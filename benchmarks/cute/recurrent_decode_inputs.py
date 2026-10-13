"""Raw decode fixtures, FP64 references and optional installed baseline adapters."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING
from typing import Any

from examples.linear.recurrent_decode import kernel_args
from examples.linear.recurrent_decode import kernel_for
import torch
import torch.nn.functional as F

import helion

if TYPE_CHECKING:
    from collections.abc import Callable


def cases() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for batch in (1, 64):
        result.append(
            {
                "name": f"gdn-decode-b{batch}-h16-hv32-d128-fp32",
                "family": "gdn-decode",
                "batch": batch,
                "tokens": 1,
                "heads": 16,
                "value_heads": 32,
                "key_dim": 128,
                "value_dim": 128,
                "implementations": [
                    "helion",
                    "flashinfer-cake",
                    "flashinfer-cute",
                    "fla",
                    "vllm-packed",
                    "sglang-packed",
                ],
                "source": "FlashInfer GDN indexed FP32 decode; Qwen grouped-head dimensions",
            }
        )
    for batch, tokens, update in ((1, 2, False), (16, 4, True)):
        result.append(
            {
                "name": f"gdn-mtp-b{batch}-t{tokens}-h16-hv32-d128-fp32",
                "family": "gdn-mtp",
                "batch": batch,
                "tokens": tokens,
                "heads": 16,
                "value_heads": 32,
                "key_dim": 128,
                "value_dim": 128,
                "program": "gdn-mtp-fp32" if tokens == 2 else "gdn-mtp-fp16",
                "update_state": update,
                "strided_inputs": True,
                "implementations": ["helion", "flashinfer-cake", "flashinfer-cute"],
                "source": "Installed FlashInfer FP32 MTP verify/update checkpoint contracts",
            }
        )
    for batch in (1, 64):
        result.append(
            {
                "name": f"kda-decode-b{batch}-h32-d128-bf16",
                "family": "kda-decode",
                "batch": batch,
                "tokens": 1,
                "heads": 32,
                "value_heads": 32,
                "key_dim": 128,
                "value_dim": 128,
                "implementations": [
                    "helion",
                    "flashinfer-cake",
                    "flashinfer-cute",
                    "fla",
                    "vllm-kda-recurrent",
                    "vllm-kda-packed",
                    "sglang-kda-packed",
                ],
                "source": "Installed recurrent KDA equal-head D128 unbounded-softplus T1",
            }
        )
    for batch, heads, value_dim, include_cake in (
        (1, 128, 64, False),
        (64, 64, 128, True),
    ):
        implementations = [
            "helion",
            "flashinfer-native",
            "mamba",
            "sglang-ssu",
            "vllm-ssu",
        ]
        if include_cake:
            implementations.insert(1, "flashinfer-cake")
        result.append(
            {
                "name": f"mamba2-decode-b{batch}-h{heads}-p{value_dim}-n128-g8-bf16",
                "family": "mamba2-decode",
                "batch": batch,
                "heads": heads,
                "value_dim": value_dim,
                "state_dim": 128,
                "groups": 8,
                "implementations": implementations,
                "source": "Trace-native Mamba H128/P64/N128/G8"
                if not include_cake
                else "Installed Cake selective-state-update P128/N128 contract",
            }
        )
    for batch in (1, 64):
        result.append(
            {
                "name": f"retention-decode-b{batch}-h8-d128-fp32",
                "family": "retention-decode",
                "batch": batch,
                "tokens": 1,
                "heads": 8,
                "value_heads": 8,
                "key_dim": 128,
                "value_dim": 128,
                "implementations": [
                    "helion",
                    "fla-recurrent-prepared",
                    "fla-chunk-prepared",
                    "fla-chunk-prepared-gamma",
                    "fla-recurrent-prepared-gamma",
                ],
                "source": "FLA retention fixed per-head decay, T1 with FP32 final state",
            }
        )
    return result


def _gdn_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    checkpoints: torch.Tensor,
    scale: float,
    tokens: int,
    update_state: bool,
    cache_states: bool,
) -> torch.Tensor:
    value_heads = v.size(2)
    head_map = torch.arange(value_heads, device=q.device) // (value_heads // q.size(2))
    query = q.double().index_select(2, head_map)
    key = k.double().index_select(2, head_map)
    query = query / ((query * query).sum(-1, keepdim=True) + 1e-06).sqrt()
    key = key / ((key * key).sum(-1, keepdim=True) + 1e-06).sqrt()
    decay = torch.exp(
        -a_log.double().exp()[None, None, :] * F.softplus(a.double() + dt_bias.double())
    )
    beta = b.double().sigmoid()
    slots = indices.long()
    current = state.index_select(0, slots).double()
    for it in range(tokens):
        current = current * decay[:, it, :, None, None]
        prediction = torch.einsum("bhvk,bhk->bhv", current, key[:, it])
        residual = (v[:, it].double() - prediction) * beta[:, it, :, None]
        current = current + torch.einsum("bhv,bhk->bhvk", residual, key[:, it])
        output[:, it].copy_(
            torch.einsum("bhvk,bhk->bhv", current, query[:, it]) * scale
        )
        if cache_states:
            checkpoints[:, it].copy_(current)
    if update_state:
        state.index_copy_(0, slots, current.to(state.dtype))
    return output


def _kda_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    heads, key_dim = (q.size(2), q.size(3))
    query, key = (q[:, 0].double(), k[:, 0].double())
    query = query / ((query * query).sum(-1, keepdim=True) + 1e-06).sqrt()
    key = key / ((key * key).sum(-1, keepdim=True) + 1e-06).sqrt()
    log_decay = -a_log.double().exp()[None, :, None] * F.softplus(
        gate[:, 0].double() + dt_bias.double().reshape(heads, key_dim)
    )
    slots = indices.long()
    current = state.index_select(0, slots).double() * log_decay.exp()[:, :, None, :]
    residual = (v[:, 0].double() - torch.einsum("bhvk,bhk->bhv", current, key)) * beta[
        :, 0
    ].double().sigmoid()[:, :, None]
    current = current + torch.einsum("bhv,bhk->bhvk", residual, key)
    output[:, 0].copy_(torch.einsum("bhvk,bhk->bhv", current, query) * scale)
    state.index_copy_(0, slots, current.to(state.dtype))
    return output


def _kda_split_reference(
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
    num_tokens: int,
    use_gate_in_kernel: bool,
    use_lower_bound: bool,
    beta_is_logit: bool,
) -> torch.Tensor:
    _kda_reference(
        q.unsqueeze(1),
        k.unsqueeze(1),
        v.unsqueeze(1),
        raw_gate.unsqueeze(1),
        beta.unsqueeze(1),
        a_log,
        dt_bias,
        state_pool,
        state_indices[:, 0],
        output.unsqueeze(1),
        scale,
    )
    return output


def _mamba_reference(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    dt_bias: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    heads = x.size(1)
    group_map = torch.arange(heads, device=x.device) // (heads // b.size(1))
    delta = F.softplus(dt.double() + dt_bias.double())
    slots = indices.long()
    current = state.index_select(0, slots).double() * torch.exp(
        delta[..., None] * a.double()
    )
    current = (
        current
        + (delta * x.double())[..., None]
        * b.double().index_select(1, group_map)[:, :, None, :]
    )
    output.copy_(
        torch.einsum("bhpn,bhn->bhp", current, c.double().index_select(1, group_map))
        + x.double() * d.double()
    )
    state.index_copy_(0, slots, current.to(state.dtype))
    return output


def _retention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor,
    final_state: torch.Tensor,
    output: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    heads = q.size(2)
    decay = 1.0 - 2.0 ** (
        -5.0 - torch.arange(heads, dtype=torch.float32, device=q.device)
    )
    current = initial_state.double() * decay.double()[None, :, None, None]
    current = current + torch.einsum(
        "bhv,bhk->bhvk", v[:, 0].double(), k[:, 0].double()
    )
    output[:, 0].copy_(torch.einsum("bhvk,bhk->bhv", current, q[:, 0].double()) * scale)
    final_state.copy_(current)
    return output


def _make_kernel(family: str, program: str | None = None) -> helion.Kernel:
    original = kernel_for(family, program)
    settings = original.settings.copy(
        autotune_benchmark_subprocess=False,
        autotune_baseline_fn=None,
        autotune_baseline_atol=0.02,
        autotune_baseline_rtol=0.01,
    )
    return helion.kernel(
        original.fn,
        settings=settings,
        cute_structural_policy=original.cute_structural_policy,
    )


def _oracle(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    mutable: tuple[int, ...],
    output_indices: tuple[int, ...],
) -> Callable[[], tuple[torch.Tensor, ...]]:
    pristine = {index: args[index].clone() for index in mutable}

    def reference() -> tuple[torch.Tensor, ...]:
        copied = list(args)
        for index, tensor in pristine.items():
            copied[index] = tensor.clone()
        fn(*copied)
        return tuple(copied[index] for index in output_indices)

    return reference


def build(case: dict[str, Any], implementation: str) -> dict[str, Any]:
    if implementation not in case["implementations"]:
        raise NotImplementedError(
            f"{implementation} has no matching {case['name']} contract"
        )
    device = torch.device(case.get("device", "cuda"))
    generator = torch.Generator(device=device).manual_seed(case.get("seed", 20261002))

    def normal(
        shape: tuple[int, ...], dtype: torch.dtype = torch.bfloat16, std: float = 0.25
    ) -> torch.Tensor:
        return torch.randn(shape, dtype=dtype, device=device, generator=generator).mul_(
            std
        )

    def activation(shape: tuple[int, ...]) -> torch.Tensor:
        if not case.get("strided_inputs"):
            return normal(shape)
        storage = normal((*shape[:-1], shape[-1] * 2))
        return storage[..., : shape[-1]]

    batch, heads = (case["batch"], case["heads"])
    family = case["family"]
    state_pool: torch.Tensor | None = None
    pool_pristine: torch.Tensor | None = None
    pool_reset: tuple[torch.Tensor, torch.Tensor] | None = None
    split_args: tuple[object, ...] | None = None
    direct_invoke: Callable[[], tuple[torch.Tensor, ...]] | None = None
    inputs: dict[str, torch.Tensor] = {}
    packed_inputs: dict[str, torch.Tensor] = {}
    kernels: dict[str, helion.Kernel] = {}
    metadata: dict[str, Any] = {
        "family": family,
        "implementation": implementation,
        "seed": case.get("seed", 20261002),
        "io_dtype": "bfloat16",
        "timed_boundary": "raw native inputs to output and required state writes",
        "reset": "pristine mutable buffers outside every timed invocation",
    }
    if family in ("gdn-decode", "gdn-mtp", "kda-decode", "retention-decode"):
        tokens, value_heads = (case["tokens"], case["value_heads"])
        key_dim, value_dim = (case["key_dim"], case["value_dim"])
        packed_decode = family in ("gdn-decode", "kda-decode")
        if packed_decode:
            qk_width = heads * key_dim
            value_width = value_heads * value_dim
            mixed_qkv = normal((batch, 2 * qk_width + value_width))
            packed_inputs["mixed_qkv"] = mixed_qkv
            row_stride = mixed_qkv.stride(0)
            q = mixed_qkv.as_strided(
                (batch, 1, heads, key_dim),
                (row_stride, row_stride, key_dim, 1),
                storage_offset=0,
            )
            k = mixed_qkv.as_strided(
                (batch, 1, heads, key_dim),
                (row_stride, row_stride, key_dim, 1),
                storage_offset=qk_width,
            )
            v = mixed_qkv.as_strided(
                (batch, 1, value_heads, value_dim),
                (row_stride, row_stride, value_dim, 1),
                storage_offset=2 * qk_width,
            )
        else:
            q, k = (
                activation((batch, tokens, heads, key_dim)),
                activation((batch, tokens, heads, key_dim)),
            )
            v = activation((batch, tokens, value_heads, value_dim))
        scale = key_dim ** (-0.5)
        state_dtype = torch.bfloat16 if family == "kda-decode" else torch.float32
        if packed_decode:
            state_pool = normal(
                (batch + 2, value_heads, value_dim, key_dim), state_dtype, 0.02
            )
            pool_pristine = state_pool.clone()
            pool_reset = (state_pool, pool_pristine)
            state = state_pool[1:-1]
            pooled_indices = torch.arange(
                1, batch + 1, dtype=torch.int32, device=device
            )
            packed_inputs.update(state_pool=state_pool, pooled_indices=pooled_indices)
        else:
            state = normal((batch, value_heads, value_dim, key_dim), state_dtype, 0.02)
        indices = torch.arange(batch, dtype=torch.int32, device=device)
        output = torch.zeros(
            (batch, tokens, value_heads, value_dim), dtype=v.dtype, device=device
        )
        metadata.update(
            state_layout="[batch,value_head,value,key] (K-last / V-major)",
            state_dtype=str(state_dtype),
            scale=scale,
        )
        inputs = {
            "q": q,
            "k": k,
            "v": v,
            "state": state,
            "indices": indices,
            "output": output,
        }
        if packed_decode:
            inputs.update(packed_inputs)
            metadata.update(
                qkv_storage="Shared contiguous [batch,q|k|v] storage; every provider receives the same strided q/k/v views",
                timed_qkv_concatenation=False,
                input_layout_copies_outside_timing=False,
                state_pool="One null prefix slot, batch active slots, one unused suffix slot; guard slots returned for correctness checks",
                state_index_mapping="Existing providers use active pool view with 0-based indices; packed providers use the full pool with indices 1..batch",
            )
    if family in ("gdn-decode", "gdn-mtp"):
        q, k, v, state, indices, output = (
            inputs[name] for name in ("q", "k", "v", "state", "indices", "output")
        )
        scale = metadata["scale"]
        tokens, value_heads = (case["tokens"], case["value_heads"])
        key_dim, value_dim = (case["key_dim"], case["value_dim"])
        a, b = (
            activation((batch, tokens, value_heads)),
            activation((batch, tokens, value_heads)),
        )
        a_log, bias = (
            normal((value_heads,), torch.float32, 0.2),
            normal((value_heads,), torch.float32, 0.1),
        )
        update = case.get("update_state", True)
        cache = family == "gdn-mtp"
        checkpoints = torch.zeros(
            (batch, tokens, value_heads, value_dim, key_dim) if cache else (1,),
            device=device,
            dtype=state.dtype,
        )
        args = (
            q,
            k,
            v,
            a,
            b,
            a_log,
            bias,
            state,
            indices,
            output,
            checkpoints,
            scale,
            tokens,
            update,
            cache,
        )
        baseline, mutable, result_indices = (
            _gdn_reference,
            (7, 9, 10),
            (9, 7, 10) if cache else (9, 7),
        )
        mutable_tensors = {7: state, 9: output, 10: checkpoints}
        result_tensors = (output, state, checkpoints) if cache else (output, state)
        inputs.update(a=a, b=b, a_log=a_log, dt_bias=bias, checkpoints=checkpoints)
        metadata.update(
            qk_l2norm=True,
            l2_epsilon=1e-06,
            gate="-exp(A_log)*softplus(a+dt_bias)",
            beta="sigmoid(raw_b)",
            head_mapping="h_v // (HV/H)",
            update_state=update,
            checkpoints=cache,
            strided_inputs=family == "gdn-decode" or case.get("strided_inputs", False),
        )
        if implementation == "helion":
            kernel = _make_kernel(family, case.get("program"))
            kernels["gdn"] = kernel

            prepared_args = kernel_args(family, args)

            def invoke() -> object:
                return kernel(*prepared_args)
        elif implementation.startswith("flashinfer"):
            module = importlib.import_module("flashinfer.gdn_decode")
            if implementation == "flashinfer-cake":
                backend = "cake_gdn"
                metadata["route"] = (
                    "strict public gated_delta_rule_decode_pretranspose backend=cake_gdn"
                )
                metadata["timed_layout_adaptation"] = (
                    "No layout copies; installed FP32 MTP accepts native strides"
                    if cache
                    else "q/k/v contiguous materialization inside run, required by Cake FP32 T1; singleton batch aliases when already contiguous"
                )

                def invoke() -> object:
                    return module.gated_delta_rule_decode_pretranspose(
                        q if cache else q.contiguous(),
                        k if cache else k.contiguous(),
                        v if cache else v.contiguous(),
                        None,
                        a_log,
                        a,
                        bias,
                        b,
                        scale=scale,
                        output=output,
                        use_qk_l2norm=True,
                        initial_state=state,
                        initial_state_indices=indices,
                        intermediate_states_buffer=checkpoints if cache else None,
                        disable_state_update=not update,
                        backend=backend,
                    )
            elif cache:
                metadata["route"] = "public gated_delta_rule_mtp FP32"

                def invoke() -> object:
                    return module.gated_delta_rule_mtp(
                        q,
                        k,
                        v,
                        state,
                        indices,
                        a_log,
                        a,
                        bias,
                        b,
                        scale=scale,
                        output=output,
                        intermediate_states_buffer=checkpoints,
                        disable_state_update=not update,
                        use_qk_l2norm=True,
                    )
            else:
                metadata["route"] = (
                    "public gated_delta_rule_decode_pretranspose backend=flashinfer"
                )

                def invoke() -> object:
                    return module.gated_delta_rule_decode_pretranspose(
                        q,
                        k,
                        v,
                        None,
                        a_log,
                        a,
                        bias,
                        b,
                        scale=scale,
                        output=output,
                        use_qk_l2norm=True,
                        initial_state=state,
                        initial_state_indices=indices,
                        backend="flashinfer",
                    )
        elif implementation in ("vllm-packed", "sglang-packed"):
            mixed_qkv = inputs["mixed_qkv"]
            pooled_indices = inputs["pooled_indices"]
            module = importlib.import_module(
                "vllm.third_party.flash_linear_attention.ops.fused_recurrent"
                if implementation == "vllm-packed"
                else "sglang.kernels.ops.attention.fla.fused_recurrent"
            )
            fn = module.fused_recurrent_gated_delta_rule_packed_decode
            metadata.update(
                route=f"installed {implementation} fused_recurrent_gated_delta_rule_packed_decode",
                launch={"BV": 32, "num_warps": 1, "num_stages": 3, "pdl": False},
                imported_module=module.__name__,
            )
            if implementation == "sglang-packed":
                metadata["beta"] = "BF16(sigmoid(raw_beta)), then FP32 accumulation"

            def invoke() -> object:
                return fn(
                    mixed_qkv=mixed_qkv,
                    a=a[:, 0],
                    b=b[:, 0],
                    A_log=a_log,
                    dt_bias=bias,
                    initial_state=state_pool,
                    ssm_state_indices=pooled_indices,
                    scale=scale,
                    use_qk_l2norm_in_kernel=True,
                    out=output,
                )
        else:
            fla_fn = importlib.import_module(
                "fla.ops.gated_delta_rule"
            ).fused_recurrent_gated_delta_rule
            metadata["route"] = (
                "public FLA fused_recurrent_gated_delta_rule; separate FP32 final-state allocation"
            )

            def fla_gdn() -> tuple[torch.Tensor, ...]:
                out, final = fla_fn(
                    q=q,
                    k=k,
                    v=v,
                    g=a,
                    beta=b,
                    A_log=a_log,
                    dt_bias=bias,
                    scale=scale,
                    initial_state=state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    use_gate_in_kernel=True,
                    use_beta_sigmoid_in_kernel=True,
                    state_v_first=True,
                )
                return (out, final)

            invoke = fla_gdn
            if family == "gdn-decode" and implementation == "fla":
                direct_invoke = fla_gdn
    elif family == "kda-decode":
        q, k, v, state, indices, output = (
            inputs[name] for name in ("q", "k", "v", "state", "indices", "output")
        )
        scale = metadata["scale"]
        key_dim = case["key_dim"]
        mixed_qkv = inputs["mixed_qkv"]
        pooled_indices = inputs["pooled_indices"]
        gate, beta = (normal((batch, 1, heads, key_dim)), normal((batch, 1, heads)))
        a_log, bias = (
            normal((heads,), torch.float32, 0.2),
            normal((heads * key_dim,), torch.float32, 0.1),
        )
        args = (q, k, v, gate, beta, a_log, bias, state, indices, output, scale)
        baseline, mutable, result_indices = (_kda_reference, (7, 9), (9, 7))
        mutable_tensors = {7: state, 9: output}
        result_tensors = (output, state)
        inputs.update(gate=gate, beta=beta, a_log=a_log, dt_bias=bias)
        metadata.update(
            qk_l2norm=True,
            l2_epsilon=1e-06,
            gate="-exp(A_log)*softplus(g+dt_bias)",
            beta="sigmoid(raw_beta)",
            update_state=True,
            gate_parameter_heads="equal query/value heads",
        )
        if implementation == "helion":
            kernel = _make_kernel(family, case.get("program"))
            kernels["kda"] = kernel
            accepted = torch.ones(batch, dtype=torch.int32, device=device)
            split_args = (
                q[:, 0],
                k[:, 0],
                v[:, 0],
                gate[:, 0],
                beta[:, 0],
                a_log,
                bias,
                state,
                indices[:, None],
                accepted,
                output[:, 0],
                scale,
                0.0,
                1,
                True,
                False,
                True,
            )
            metadata["helion_body"] = "examples.linear.recurrent_decode.kda_decode"

            prepared_args = kernel_args(family, split_args)

            def invoke() -> object:
                return kernel(*prepared_args)
        elif implementation.startswith("flashinfer"):
            fn = importlib.import_module(
                "flashinfer.kda_kernels.recurrent_kda"
            ).run_recurrent_kda
            backend = "cake" if implementation == "flashinfer-cake" else "cute-dsl"
            metadata["route"] = (
                f"run_recurrent_kda backend={backend}; explicit cu_seqlens omitted"
            )

            def invoke() -> object:
                return fn(
                    q=q,
                    k=k,
                    v=v,
                    g=gate,
                    beta=beta,
                    A_log=a_log,
                    dt_bias=bias,
                    scale=scale,
                    initial_state=state,
                    output_final_state=False,
                    use_qk_l2norm_in_kernel=True,
                    use_gate_in_kernel=True,
                    ssm_state_indices=indices,
                    output=output,
                    beta_is_logit=True,
                    backend=backend,
                )
        elif implementation in ("vllm-kda-recurrent", "vllm-kda-packed"):
            module = importlib.import_module(
                "vllm.models.kimi_k3.nvidia.ops.third_party.kda.fused_recurrent"
            )
            raw_gate = gate.transpose(0, 1)
            raw_beta = beta.transpose(0, 1)
            flat_output = output.transpose(0, 1)
            cu_seqlens = torch.arange(batch + 1, dtype=torch.int32, device=device)
            inputs["cu_seqlens"] = cu_seqlens
            if implementation == "vllm-kda-recurrent":
                fn = module.fused_recurrent_kda
                q_flat, k_flat, v_flat = (
                    tensor.transpose(0, 1) for tensor in (q, k, v)
                )
                metadata["launch"] = {
                    "BV": 4
                    if heads * batch <= 48
                    else 8
                    if heads * batch <= 96
                    else 16
                    if heads * batch <= 192
                    else 8,
                    "num_warps": 1,
                    "num_stages": 4 if heads * batch <= 48 else 3,
                    "pdl": module.current_platform.is_arch_support_pdl(),
                }

                def invoke() -> object:
                    return fn(
                        q=q_flat,
                        k=k_flat,
                        v=v_flat,
                        raw_g=raw_gate,
                        raw_beta=raw_beta,
                        A_log=a_log,
                        dt_bias=bias,
                        lower_bound=None,
                        initial_state=state_pool,
                        cu_seqlens=cu_seqlens,
                        ssm_state_indices=pooled_indices,
                        out=flat_output,
                        fuse_gate=True,
                    )
            else:
                fn = module.fused_recurrent_kda_packed_decode
                metadata["launch"] = {
                    "BV": 32,
                    "num_warps": 4,
                    "num_stages": 2,
                    "pdl": module.current_platform.is_arch_support_pdl(),
                }

                def invoke() -> object:
                    return fn(
                        mixed_qkv=mixed_qkv,
                        raw_g=raw_gate,
                        raw_beta=raw_beta,
                        A_log=a_log,
                        dt_bias=bias,
                        lower_bound=None,
                        initial_state=state_pool,
                        state_indices=pooled_indices,
                        scale=scale,
                        out=flat_output,
                    )

            metadata.update(
                route=f"installed vLLM {fn.__name__}", imported_module=module.__name__
            )
        elif implementation == "sglang-kda-packed":
            module = importlib.import_module(
                "sglang.kernels.ops.attention.fla.fused_recurrent"
            )
            cuda_module = importlib.import_module(
                "sglang.kernels.ops.attention.kda_packed_decode"
            )
            raw_gate = gate.view(batch, heads * key_dim)
            raw_beta = beta[:, 0]
            cuda_covered = cuda_module.covered(
                mixed_qkv,
                raw_gate,
                raw_beta,
                a_log,
                bias,
                state_pool,
                output,
                pooled_indices,
                heads,
            )
            if cuda_covered:
                raise NotImplementedError(
                    "This adapter loads SGLang's Triton KDA route; the exact CUDA coverage predicate admitted another contract"
                )
            fn = module.fused_recurrent_kda_packed_decode
            metadata.update(
                route="installed SGLang fused_recurrent_kda_packed_decode; native Triton fallback",
                launch={"BV": 32, "num_warps": 1, "num_stages": 3, "pdl": False},
                cuda_fast_path_covered=False,
                cuda_fast_path_exclusion="Original covered() requires FP32 initial_state; this case uses BF16",
                imported_module=module.__name__,
            )

            def invoke() -> object:
                return fn(
                    mixed_qkv=mixed_qkv,
                    a=raw_gate,
                    b=raw_beta,
                    A_log=a_log,
                    dt_bias=bias,
                    scale=scale,
                    initial_state=state_pool,
                    out=output,
                    ssm_state_indices=pooled_indices,
                    use_qk_l2norm_in_kernel=True,
                    lower_bound=None,
                )
        else:
            fn = importlib.import_module(
                "fla.ops.kda.fused_recurrent"
            ).fused_recurrent_kda_fwd
            metadata["route"] = (
                "installed FLA serving fused_recurrent_kda_fwd; inplace_final_state=True and BF16 state"
            )

            def invoke() -> object:
                return fn(
                    q=q,
                    k=k,
                    v=v,
                    g=gate,
                    beta=beta,
                    A_log=a_log,
                    dt_bias=bias,
                    scale=scale,
                    initial_state=state,
                    inplace_final_state=True,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    use_gate_in_kernel=True,
                    use_beta_sigmoid_in_kernel=True,
                    state_v_first=True,
                    ssm_state_indices=indices,
                    out=output,
                )
    elif family == "mamba2-decode":
        value_dim, state_dim, groups = (
            case["value_dim"],
            case["state_dim"],
            case["groups"],
        )
        state = normal((batch, heads, value_dim, state_dim), torch.bfloat16, 0.02)
        if implementation == "vllm-ssu":
            state_pool = torch.full(
                (batch + 2, heads, value_dim, state_dim),
                0.125,
                dtype=state.dtype,
                device=device,
            )
            state_pool[-1].fill_(-0.125)
            state_pool[1:-1].copy_(state)
            pool_pristine = state_pool.clone()
            pool_reset = (state_pool, pool_pristine)
            state = state_pool[1:-1]
            pooled_indices = torch.arange(
                1, batch + 1, dtype=torch.int64, device=device
            )
            packed_inputs.update(state_pool=state_pool, pooled_indices=pooled_indices)
        x = normal((batch, heads, value_dim))
        dt = normal((batch, heads, 1), torch.float32).expand(batch, heads, value_dim)
        a = (-normal((heads, 1, 1), torch.float32).abs() - 0.5).expand(
            heads, value_dim, state_dim
        )
        b, c = (normal((batch, groups, state_dim)), normal((batch, groups, state_dim)))
        d = normal((heads, 1), torch.float32).expand(heads, value_dim)
        bias = (normal((heads, 1), torch.float32, 0.1) - 2.0).expand(heads, value_dim)
        indices = torch.arange(batch, dtype=torch.int64, device=device)
        output = torch.zeros_like(x)
        args = (state, x, dt, a, b, c, d, bias, indices, output)
        baseline, mutable, result_indices = (_mamba_reference, (0, 9), (9, 0))
        mutable_tensors = {0: state, 9: output}
        result_tensors = (output, state)
        inputs = {
            "state": state,
            "x": x,
            "dt": dt,
            "A": a,
            "B": b,
            "C": c,
            "D": d,
            "dt_bias": bias,
            "indices": indices,
            "output": output,
        }
        metadata.update(
            state_layout="[batch,head,value,state]",
            state_dtype="torch.bfloat16",
            dt_dtype="torch.float32",
            dt_softplus=True,
            tied_dt_A_D_bias=True,
            groups=groups,
            has_D=True,
            has_z=False,
            update_state=True,
        )
        if implementation == "helion":
            kernel = _make_kernel(family, case.get("program"))
            kernels["mamba"] = kernel

            prepared_args = kernel_args(family, args)

            def invoke() -> object:
                return kernel(*prepared_args)
        elif implementation.startswith("flashinfer"):
            fn = importlib.import_module(
                "flashinfer.mamba.selective_state_update"
            ).selective_state_update
            backend = "cake" if implementation == "flashinfer-cake" else "flashinfer"
            metadata["route"] = f"public selective_state_update backend={backend}"
            metadata["backend_language"] = (
                "CUDA/Cake" if backend == "cake" else "FlashInfer native CUDA"
            )
            if backend == "cake":
                cake_module = importlib.import_module(
                    "flashinfer.jit.mamba.cake_selective_state_update"
                )
                original = cake_module.try_cake_selective_state_update

                def require_cake(*positional: object, **kwargs: object) -> bool:
                    admitted = original(*positional, **kwargs)
                    if not admitted:
                        raise NotImplementedError(
                            "Cake selective_state_update declined this contract; fallback is not benchmarked"
                        )
                    metadata["cake_dispatch_confirmed"] = True
                    return True

                vars(cake_module)["try_cake_selective_state_update"] = require_cake

            def invoke() -> object:
                return fn(
                    state=state,
                    x=x,
                    dt=dt,
                    A=a,
                    B=b,
                    C=c,
                    D=d,
                    dt_bias=bias,
                    dt_softplus=True,
                    state_batch_indices=indices,
                    out=output,
                    backend=backend,
                )
        elif implementation == "sglang-ssu":
            module = importlib.import_module(
                "sglang.kernels.ops.mamba.triton_ops.mamba_ssm"
            )
            fn = module.selective_state_update
            metadata.update(
                route="installed SGLang selective_state_update",
                launch={
                    "BLOCK_SIZE_M": 8,
                    "num_warps": 1,
                    "pdl": module.is_arch_support_pdl(),
                },
                imported_module=module.__name__,
            )

            def invoke() -> object:
                return fn(
                    state=state,
                    x=x,
                    dt=dt,
                    A=a,
                    B=b,
                    C=c,
                    D=d,
                    dt_bias=bias,
                    dt_softplus=True,
                    state_batch_indices=indices,
                    out=output,
                )
        elif implementation == "vllm-ssu":
            module = importlib.import_module(
                "vllm.model_executor.layers.mamba.ops.mamba_ssm"
            )
            fn = module.selective_state_update
            is_blackwell = (
                device.type == "cuda"
                and module.current_platform.is_device_capability_family(100)
            )
            launch: dict[str, Any] = {
                "policy": "original try_get_optimal_ssm_config; JSON lookup then native fallback",
                "is_blackwell": is_blackwell,
                "resolved_on_cuda": device.type == "cuda",
            }
            if device.type == "cuda":
                cache_dtype = str(state.dtype).removeprefix("torch.")
                configs = module.get_ssm_configs(value_dim, state_dim, cache_dtype)
                block_size_m, num_warps = module.try_get_optimal_ssm_config(
                    value_dim, state_dim, batch, heads, cache_dtype, is_blackwell
                )
                launch.update(
                    BLOCK_SIZE_M=block_size_m,
                    num_warps=num_warps,
                    configuration_source="original JSON lookup"
                    if configs
                    else "original fallback heuristic; matching JSON unavailable",
                    device_name=module.get_ssm_device_name(),
                    config_filename=module.get_ssm_config_file_name(
                        value_dim,
                        state_dim,
                        module._canonical_cache_dtype(cache_dtype),
                        module.get_ssm_device_name(),
                    ),
                    bundled_config_directory=module._CONFIGS_DIR,
                    user_config_directory=module.envs.VLLM_TUNED_CONFIG_FOLDER,
                )
            state_pool = packed_inputs["state_pool"]
            pooled_indices = packed_inputs["pooled_indices"]
            inputs.update(packed_inputs)
            metadata.update(
                route="installed vLLM Triton selective_state_update",
                launch=launch,
                null_block_id=module.NULL_BLOCK_ID,
                state_index_mapping="Same active state values in slots1..batch of a new guarded pool; original public null-block constant retained",
                state_pool="Additional prefix/suffix guards; active state copy occurs once during input preparation, with no timed copy or extra RNG draws",
                stochastic_rounding=False,
                imported_module=module.__name__,
            )

            def invoke() -> object:
                return fn(
                    state=state_pool,
                    x=x,
                    dt=dt,
                    A=a,
                    B=b,
                    C=c,
                    D=d,
                    dt_bias=bias,
                    dt_softplus=True,
                    state_batch_indices=pooled_indices,
                    out=output,
                    is_blackwell=is_blackwell,
                )
        else:
            fn = importlib.import_module(
                "mamba_ssm.ops.triton.selective_state_update"
            ).selective_state_update
            metadata["route"] = "installed Mamba Triton selective_state_update"

            def mamba_run() -> tuple[torch.Tensor, ...]:
                out = fn(
                    state,
                    x,
                    dt,
                    a,
                    b,
                    c,
                    D=d,
                    dt_bias=bias,
                    dt_softplus=True,
                    state_batch_indices=indices,
                )
                return (out, state)

            invoke = mamba_run
            if implementation == "mamba":
                direct_invoke = mamba_run
    elif family == "retention-decode":
        q, k, v, state, indices, output = (
            inputs[name] for name in ("q", "k", "v", "state", "indices", "output")
        )
        scale = metadata["scale"]
        final_state = torch.zeros_like(state)
        args = (q, k, v, state, final_state, output, scale)
        baseline, mutable, result_indices = (_retention_reference, (4, 5), (5, 4))
        mutable_tensors = {4: final_state, 5: output}
        result_tensors = (output, final_state)
        inputs["final_state"] = final_state
        metadata.update(
            qk_l2norm=False,
            decay="1-2**(-5-head), FP32",
            initial_state_readonly=True,
            final_state="separate FP32 output",
        )
        if implementation == "helion":
            kernel = _make_kernel(family, case.get("program"))
            kernels["retention"] = kernel

            prepared_args = kernel_args(family, args)

            def invoke() -> object:
                return kernel(*prepared_args)
        else:
            fn = importlib.import_module("fla.ops.retention").fused_recurrent_retention
            metadata["route"] = (
                "public FLA fused_recurrent_retention, state_v_first=True"
            )

            def retention_run() -> tuple[torch.Tensor, ...]:
                return fn(
                    q=q,
                    k=k,
                    v=v,
                    scale=scale,
                    initial_state=state,
                    output_final_state=True,
                    state_v_first=True,
                )

            invoke = retention_run
            if implementation == "fla":
                direct_invoke = retention_run
    else:
        raise NotImplementedError(family)
    pristine = {index: tensor.clone() for index, tensor in mutable_tensors.items()}

    def reset() -> None:
        for index, tensor in pristine.items():
            mutable_tensors[index].copy_(tensor)
        if pool_reset is not None:
            pool, original_pool = pool_reset
            pool.copy_(original_pool)

    def run() -> tuple[torch.Tensor, ...]:
        if direct_invoke is not None:
            result = tuple(direct_invoke())
        else:
            invoke()
            result = result_tensors
        if state_pool is not None:
            return (*result, state_pool[:1], state_pool[-1:])
        return result

    oracle = _oracle(baseline, args, mutable, result_indices)

    def reference() -> tuple[torch.Tensor, ...]:
        expected = oracle()
        if pool_pristine is not None:
            return (*expected, pool_pristine[:1].clone(), pool_pristine[-1:].clone())
        return expected

    return {
        "run": run,
        "reset": reset,
        "reference": reference,
        "kernels": kernels,
        "metadata": metadata,
        "inputs": inputs,
        "atol": 0.02,
        "rtol": 0.01,
        "objective_binding": {
            "canonical_args": args,
            "kernel_args": split_args if split_args is not None else args,
            "mutable_indices": mutable,
            "result_indices": result_indices,
            "state_pool": state_pool,
            "baseline": baseline,
            "kernel_reference": _kda_split_reference
            if family == "kda-decode"
            else baseline,
        },
    }


_BUILD_CANONICAL = build
RETENTION_ROUTES = {
    "fla-recurrent-prepared": ("fused_recurrent", "fused_recurrent_simple_gla", False),
    "fla-chunk-prepared": ("chunk", "chunk_simple_gla", False),
    "fla-chunk-prepared-gamma": ("chunk", "chunk_simple_gla", True),
    "fla-recurrent-prepared-gamma": (
        "fused_recurrent",
        "fused_recurrent_simple_gla",
        True,
    ),
}


def build(case: dict[str, Any], implementation: str) -> dict[str, Any]:
    if implementation not in RETENTION_ROUTES:
        return _BUILD_CANONICAL(case, implementation)
    assert case["family"] == "retention-decode"
    fixture = _BUILD_CANONICAL(dict(case, implementations=["fla"]), "fla")
    inputs = fixture["inputs"]
    q, k, v, state = (inputs[name] for name in ("q", "k", "v", "state"))
    suffix, callee, prepared_gamma = RETENTION_ROUTES[implementation]
    fn = getattr(importlib.import_module(f"fla.ops.retention.{suffix}"), callee)
    two = q.new_tensor(2.0, dtype=torch.float)
    head_index = q.new_tensor(range(q.shape[2]), dtype=torch.float)
    inputs.update(retention_constant_two=two, retention_head_index=head_index)
    gamma = (1 - two.pow(-5.0 - head_index)).log() if prepared_gamma else None
    if prepared_gamma:
        inputs["retention_prepared_gamma"] = gamma
    scale = fixture["metadata"]["scale"]

    def run() -> tuple[torch.Tensor, torch.Tensor]:
        g_gamma = gamma if prepared_gamma else (1 - two.pow(-5.0 - head_index)).log()
        if suffix == "chunk":
            output, final = fn(
                q=q,
                k=k,
                v=v,
                scale=scale,
                g_gamma=g_gamma,
                initial_state=state.transpose(-2, -1).contiguous(),
                output_final_state=True,
                cu_seqlens=None,
                chunk_size=None,
            )
            return output, final.transpose(-2, -1).contiguous()
        return fn(
            q=q,
            k=k,
            v=v,
            g_gamma=g_gamma,
            scale=scale,
            initial_state=state,
            output_final_state=True,
            reverse=False,
            cu_seqlens=None,
            state_v_first=True,
        )

    fixture["run"] = run
    fixture["metadata"].update(
        implementation=implementation,
        route=f"installed FLA {callee}",
        gamma_prepared=prepared_gamma,
        prepared_gamma_expression="(1 - two.pow(-5.0 - head_index)).log()",
        timed_state_layout_conversion=suffix == "chunk",
        gamma_readonly=prepared_gamma,
        outside_timing="FP32 device literals, and fixed per-head gamma only for -gamma routes",
    )
    return fixture
