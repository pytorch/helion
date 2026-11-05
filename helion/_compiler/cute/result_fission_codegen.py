"""Emit the result-fission bundle: native producer GEMM, then SIMT consumer.

Both stage kernels are compiled in private child environments (like the split-K
workspace producer): the producer from a private 2-D GEMM template over the
original operand and identity-output tensors, the consumer from the original
root's own epilogue statements with the accumulator replaced by a load of the
identity output.  No user function is retraced, no tensor is allocated, and
the outer host program keeps its original metadata, preludes and assertions.
Both launches are ordinary (non-PDL) launches on the caller's stream.
"""

from __future__ import annotations

import ast
import hashlib
import symtable
from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import exc
from ... import language as hl
from ...exc import InvalidConfig
from ...runtime.config import Config
from ..ast_extension import ExtendedAST
from ..ast_extension import LoopType
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..inductor_lowering_extra import patch_inductor_lowerings
from ..kernel_compiler import KernelCompiler
from ..output_header import get_needed_import_lines
from ..variable_origin import ArgumentOrigin
from .cute_mma import analyze_cute_mma_node
from .materialized_fission_codegen import _empty_graph
from .materialized_fission_codegen import _module_source
from .memory_ops import _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY as _ALIGNMENT_KEY
from .memory_ops import _bound_vec_alignment_signature
from .memory_ops import tensor_has_specialized_tma_alignment
from .pipeline_smem import analyze_pipeline_smem_facts
from .result_fission import CONSUMER_FUNCTION
from .result_fission import analyze_result_fission
from .result_fission import consumer_function
from .result_fission_config import CONSUMER_BLOCK_SIZES_KEY
from .result_fission_config import CONSUMER_NUM_THREADS_KEY
from .result_fission_config import CONSUMER_VECTOR_WIDTH_KEY
from .result_fission_config import MAX_PRODUCER_AB_STAGES
from .result_fission_config import PRODUCER_AB_STAGES_KEY
from .result_fission_config import PRODUCER_ACC_STAGES_KEY
from .result_fission_config import PRODUCER_BLOCK_SIZES_KEY
from .result_fission_config import PRODUCER_C_STAGES_KEY
from .result_fission_config import PRODUCER_CLUSTER_M_KEY
from .result_fission_config import PRODUCER_CTA_GROUP_KEY
from .result_fission_config import PRODUCER_PID_TYPE_KEY
from .result_fission_config import ProducerAdmission
from .result_fission_config import ProducerSchedule
from .result_fission_config import producer_schedule_keys
from .result_fission_config import producer_schedules
from .split_k_workspace import _shape
from .split_k_workspace import _strides
from .tcgen05_config import CuteTcgen05Config

if TYPE_CHECKING:
    import types

    from ..host_function import HostFunction
    from .result_fission import ResultFissionProof


PRODUCER_FUNCTION = "_result_fission_producer"


def _result_fission_producer(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    for tile_m, tile_n in hl.tile([out.size(0), out.size(1)]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(a.size(1)):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


def _strict(requested: Config, normalized: Config, stage: str) -> None:
    changed = {
        key: (value, normalized.config.get(key))
        for key, value in requested.config.items()
        if normalized.config.get(key) != value
    }
    if changed:
        raise exc.BackendUnsupported(
            "cute",
            f"result-fission {stage} config not admitted as requested: {changed}",
        )


def _lowers_to_tcgen05(
    proof: ResultFissionProof, normalized: Config, device: torch.device
) -> bool:
    """The MMA lowering's own implementation choice for the producer GEMM
    (``_choose_mma_impl``, which declines TCGEN05 for an AB ring past its
    shared-memory check): a normalized config can still lower to SIMT."""
    from .cute_mma import _choose_mma_impl

    bm, bn, bk = cast("list[int]", normalized.config["block_sizes"])
    return (
        _choose_mma_impl(
            proof.operand_dtype,
            bm=bm,
            bn=bn,
            bk=bk,
            config=normalized,
            input_device=device,
        )
        == "tcgen05"
    )


def _require_native_producer(source: str) -> None:
    """The emitted producer must be the native TCGEN05 GEMM with its one TMA
    store ring -- never a silent SIMT lowering of the same template."""
    if "tcgen05" not in source or source.count("PipelineTmaStore.create(") != 1:
        raise exc.BackendUnsupported(
            "cute",
            "result-fission producer did not lower to the native TCGEN05 GEMM "
            "with one TMA store ring",
        )


def _require_pdl_free(stage: str, source: str, plans: list[dict[str, object]]) -> None:
    """Both stages must be ordinary launches: no griddepcontrol intrinsic in
    the emitted body, no programmatic-dependent launch flag on the kernel or
    in its wrapper plans.  A config's ``cute_pdl=False`` alone is not
    evidence (a lowering may still emit the intrinsics)."""
    if (
        "griddepcontrol" in source
        or "_helion_cute_use_pdl" in source
        or "'use_pdl': True" in source
        or '"use_pdl": true' in source
        or any(plan.get("use_pdl") for plan in plans)
    ):
        raise exc.BackendUnsupported(
            "cute",
            f"result-fission {stage} would launch with PDL; the schedule "
            "requires ordinary same-stream launches",
        )


def producer_config(config: Config) -> Config:
    values = config.config
    blocks = cast("list[int]", values[PRODUCER_BLOCK_SIZES_KEY])
    return Config.from_dict(
        {
            "block_sizes": list(blocks),
            "pid_type": values[PRODUCER_PID_TYPE_KEY],
            "tcgen05_ab_stages": values[PRODUCER_AB_STAGES_KEY],
            "tcgen05_acc_stages": values[PRODUCER_ACC_STAGES_KEY],
            "tcgen05_c_stages": values[PRODUCER_C_STAGES_KEY],
            "tcgen05_cluster_m": values[PRODUCER_CLUSTER_M_KEY],
            "tcgen05_cluster_n": 1,
            "tcgen05_cta_group": values[PRODUCER_CTA_GROUP_KEY],
            "cute_pdl": False,
        }
    )


def _child(owner: CompileEnvironment) -> CompileEnvironment:
    child = CompileEnvironment(
        owner.device,
        owner.settings.copy(static_shapes=True),
        index_dtype=owner.index_dtype,
        is_distributed=False,
    )
    child.config_spec.num_sm = owner.config_spec.num_sm
    return child


def _compile_child(
    owner: CompileEnvironment,
    child: CompileEnvironment,
    fn: types.FunctionType,
    metadata: list[tuple[str, tuple[int, ...], tuple[int, ...], torch.dtype]],
    *,
    tma_inputs: bool,
) -> tuple[HostFunction, list[torch.Tensor]]:
    from ...runtime.kernel import _maybe_skip_dtype_check_in_meta_registrations

    arguments = [
        torch.empty_strided(shape, strides, dtype=dtype, device=child.device)
        for _, shape, strides, dtype in metadata
    ]
    for (name, *_), tensor in zip(metadata, arguments, strict=True):
        child.input_sources[tensor] = ArgumentOrigin(name).to_source()
    if tma_inputs:
        # Every producer tensor inherits the owner's cache-key-backed
        # TensorMap base/stride proof checked by the caller.
        child.cute_proven_tma_inputs.update(arguments)
    compiler = KernelCompiler(child)
    with _maybe_skip_dtype_check_in_meta_registrations(), patch_inductor_lowerings():
        host = compiler.compile(fn, list(arguments), {})
    return host, arguments


def producer_admission(
    proof: ResultFissionProof, owner: CompileEnvironment
) -> ProducerAdmission:
    """For every statically legal producer schedule, the deepest AB ring the
    compiled producer child admits exactly as requested AND its MMA lowering
    still lowers to TCGEN05 (binary search over the depth; the tests check
    monotonicity on the full grid and the emitted kernels on samples)."""
    capacity = CuteTcgen05Config.per_cta_smem_capacity_bytes(owner.device)
    child = _child(owner)
    table: dict[ProducerSchedule, int] = {}
    with owner.suspend(), child:
        host, _ = _compile_child(
            owner,
            child,
            cast("types.FunctionType", _result_fission_producer),
            _producer_metadata(proof),
            tma_inputs=True,
        )
        with host:
            if not _register_producer_facts(child, host, capacity):
                return ProducerAdmission({})

            def admits(schedule: ProducerSchedule, stages: int) -> bool:
                requested = producer_config(
                    Config.from_dict(producer_schedule_keys(schedule, stages))
                )
                try:
                    normalized = child.config_spec.normalized_config(requested)
                    _strict(requested, normalized, "producer")
                except (InvalidConfig, exc.BackendUnsupported):
                    return False
                return _lowers_to_tcgen05(proof, normalized, owner.device)

            for schedule in producer_schedules():
                if not admits(schedule, 1):
                    table[schedule] = 0
                    continue
                low, high = 1, MAX_PRODUCER_AB_STAGES
                while low < high:
                    middle = (low + high + 1) // 2
                    if admits(schedule, middle):
                        low = middle
                    else:
                        high = middle - 1
                table[schedule] = low
    return ProducerAdmission(table)


def _producer_metadata(
    proof: ResultFissionProof,
) -> list[tuple[str, tuple[int, ...], tuple[int, ...], torch.dtype]]:
    return [
        ("a", (proof.m, proof.k), proof.lhs_strides, proof.operand_dtype),
        ("b", (proof.k, proof.n), proof.rhs_strides, proof.operand_dtype),
        ("out", (proof.m, proof.n), (proof.n, 1), torch.float32),
    ]


def _register_producer_facts(
    child: CompileEnvironment, host: HostFunction, capacity: int
) -> bool:
    """One native MMA, its pipeline SMEM facts (one C output), TCGEN05 on."""
    candidates = [
        (node, candidate)
        for info in host.device_ir.graphs
        for node in info.graph.nodes
        if (candidate := analyze_cute_mma_node(node, device_ir=host.device_ir))
        is not None
    ]
    if len(candidates) != 1:
        return False
    node, candidate = candidates[0]
    facts = analyze_pipeline_smem_facts(
        candidate, node, host.device_ir.graphs, capacity_bytes=capacity
    )
    if facts is None:
        return False
    child.config_spec._cute_tcgen05_config.register_pipeline_smem_facts(facts)
    return bool(child.config_spec.cute_tcgen05_search_enabled)


def _producer_source(
    proof: ResultFissionProof, owner: CompileEnvironment, config: Config
) -> tuple[str, list[dict[str, object]], dict[str, object]]:
    capacity = CuteTcgen05Config.per_cta_smem_capacity_bytes(owner.device)
    child = _child(owner)
    with owner.suspend(), child:
        host, _ = _compile_child(
            owner,
            child,
            cast("types.FunctionType", _result_fission_producer),
            _producer_metadata(proof),
            tma_inputs=True,
        )
        with host:
            if not _register_producer_facts(child, host, capacity):
                raise exc.BackendUnsupported(
                    "cute",
                    "result-fission producer is not a proved native TCGEN05 GEMM",
                )
            requested = producer_config(config)
            try:
                normalized = child.config_spec.normalized_config(requested)
            except InvalidConfig as error:
                raise exc.BackendUnsupported(
                    "cute", f"result-fission producer config not admitted: {error}"
                ) from error
            _strict(requested, normalized, "producer")
            if not _lowers_to_tcgen05(proof, normalized, owner.device):
                raise exc.BackendUnsupported(
                    "cute",
                    "result-fission producer config would not lower to TCGEN05",
                )
            from ..generate_ast import generate_ast

            module = generate_ast(host, normalized, False)
            source = _module_source(module)
            _require_native_producer(source)
            return (
                source,
                list(child.cute_resolved_wrapper_plans),
                dict(normalized.config),
            )


def _keys_of(config: Config) -> dict[str, object] | None:
    values = config.config
    blocks = values.get("block_sizes")
    if not (isinstance(blocks, list) and len(blocks) == 3):
        return None
    return {
        PRODUCER_BLOCK_SIZES_KEY: list(blocks),
        PRODUCER_PID_TYPE_KEY: values.get("pid_type", "flat"),
        PRODUCER_AB_STAGES_KEY: values.get("tcgen05_ab_stages", 2),
        PRODUCER_ACC_STAGES_KEY: values.get("tcgen05_acc_stages", 2),
        PRODUCER_C_STAGES_KEY: values.get("tcgen05_c_stages", 2),
        PRODUCER_CLUSTER_M_KEY: values.get("tcgen05_cluster_m", 1),
        PRODUCER_CTA_GROUP_KEY: values.get("tcgen05_cta_group", "auto"),
    }


def producer_keys_of(config: Config) -> dict[str, object] | None:
    """Producer keys naming a native TCGEN05 seed's GEMM schedule."""
    values = config.config
    blocks = values.get("block_sizes")
    if (
        not (isinstance(blocks, list) and len(blocks) == 3)
        or values.get("cute_collective_mma", False) is not False
        or "tcgen05_ab_stages" not in values
        or values.get("tcgen05_cluster_n", 1) != 1
    ):
        return None
    return _keys_of(config)


def admitted_producer_keys(
    proof: ResultFissionProof,
    owner: CompileEnvironment,
    candidates: list[dict[str, object]],
) -> list[dict[str, object]]:
    """The candidates the producer's own (one C output) normalizer admits
    exactly as requested, in order."""
    capacity = CuteTcgen05Config.per_cta_smem_capacity_bytes(owner.device)
    child = _child(owner)
    with owner.suspend(), child:
        host, _ = _compile_child(
            owner,
            child,
            cast("types.FunctionType", _result_fission_producer),
            _producer_metadata(proof),
            tma_inputs=True,
        )
        with host:
            if not _register_producer_facts(child, host, capacity):
                return []
            from ..generate_ast import generate_ast

            result: list[dict[str, object]] = []
            for keys in candidates:
                requested = producer_config(Config.from_dict(dict(keys)))
                try:
                    normalized = child.config_spec.normalized_config(requested)
                    _strict(requested, normalized, "producer")
                    if not _lowers_to_tcgen05(proof, normalized, owner.device):
                        continue
                    # The emitted body, not the config, decides PDL.
                    child.cute_resolved_wrapper_plans = []
                    module = generate_ast(host, normalized, False)
                    source = _module_source(module)
                    _require_native_producer(source)
                    _require_pdl_free(
                        "producer", source, list(child.cute_resolved_wrapper_plans)
                    )
                except (InvalidConfig, exc.BackendUnsupported):
                    continue
                result.append(keys)
            return result


def consumer_config(config: Config, base: Config) -> Config:
    blocks = cast("list[int]", config.config[CONSUMER_BLOCK_SIZES_KEY])
    threads = cast("list[int]", config.config[CONSUMER_NUM_THREADS_KEY])
    vector = cast("int", config.config[CONSUMER_VECTOR_WIDTH_KEY])
    values: dict[str, object] = {
        "block_sizes": list(blocks),
        # N, the stride-1 output axis, on threadIdx.x: coalesced lanes.
        "loop_orders": [[1, 0]],
        "pid_type": "flat",
        "cute_pdl": False,
    }
    if vector > 1:
        values["num_threads"] = list(threads)
        values["cute_vector_widths"] = [1, vector]
    if "cute_exact_div_form" in base.config:
        # Plain IEEE division, the form the fused epilogue renders.
        values["cute_exact_div_form"] = "plain"
    return Config.from_dict(values)


def _consumer_source(
    proof: ResultFissionProof,
    owner: CompileEnvironment,
    host: HostFunction,
    config: Config,
) -> tuple[str, list[dict[str, object]], dict[str, object]]:
    metadata: list[tuple[str, tuple[int, ...], tuple[int, ...], torch.dtype]] = []
    for name in proof.consumer_params:
        tensor = host.params.arguments[name]
        assert isinstance(tensor, torch.Tensor)
        shape, strides = _shape(owner, tensor), _strides(owner, tensor)
        if shape is None or strides is None:
            raise exc.BackendUnsupported(
                "cute", f"result-fission consumer argument {name} lacks static layout"
            )
        metadata.append((name, shape, strides, tensor.dtype))
    vector = cast("int", config.config[CONSUMER_VECTOR_WIDTH_KEY])
    owner_alignment: dict[str, object] = {}
    if vector > 1:
        # Vector IO reads cache-key-backed alignment facts. The consumer's
        # tensors are the owner's own arguments, whose bound kernel key
        # already carries their alignment signatures: transfer exactly those.
        for name in proof.consumer_params:
            tensor = host.params.arguments[name]
            assert isinstance(tensor, torch.Tensor)
            signature = _bound_vec_alignment_signature(
                owner, owner.tensor_input_source(tensor)
            )
            if signature is None:
                raise exc.BackendUnsupported(
                    "cute",
                    f"result-fission vector consumer lacks the owner's alignment facts for {name}",
                )
            owner_alignment[name] = signature
    child = _child(owner)
    fn = consumer_function(proof.consumer_source)
    with owner.suspend(), child:
        consumer_host, arguments = _compile_child(
            owner, child, fn, metadata, tma_inputs=False
        )
        with consumer_host:
            if child.config_spec.cute_tcgen05_search_enabled:
                raise exc.BackendUnsupported(
                    "cute", "result-fission consumer must be a pure SIMT kernel"
                )
            if owner_alignment:
                specialization = child.runtime_input_specializations.get(_ALIGNMENT_KEY)
                by_source = {
                    child.tensor_input_source(tensor): name
                    for (name, *_), tensor in zip(metadata, arguments, strict=True)
                }
                if specialization is None or any(
                    by_source.get(source) not in owner_alignment
                    for source in specialization.sources
                ):
                    raise exc.BackendUnsupported(
                        "cute", "result-fission vector consumer alignment transfer"
                    )
                child.bound_runtime_input_specialization_results = {
                    **child.bound_runtime_input_specialization_results,
                    _ALIGNMENT_KEY: tuple(
                        owner_alignment[by_source[source]]
                        for source in specialization.sources
                    ),
                }
            requested = consumer_config(config, child.config_spec.default_config())
            try:
                normalized = child.config_spec.normalized_config(requested)
            except InvalidConfig as error:
                raise exc.BackendUnsupported(
                    "cute", f"result-fission consumer config not admitted: {error}"
                ) from error
            _strict(requested, normalized, "consumer")
            from ..generate_ast import generate_ast

            module = generate_ast(consumer_host, normalized, False)
            source = _module_source(module)
            if vector > 1 and f"VectorType.get([{vector}]" not in source:
                # A vector layout whose views the cache-key facts cannot
                # prove aligned renders scalar IO: refuse the mislabeled
                # config rather than run a different layout.
                raise exc.BackendUnsupported(
                    "cute",
                    "result-fission vector consumer not admitted for these views",
                )
            return (
                source,
                list(child.cute_resolved_wrapper_plans),
                dict(normalized.config),
            )


def generate_result_fission(
    func: HostFunction, config: Config, emit_repro_caller: bool
) -> ast.Module:
    from ..generate_ast import GenerateAST
    from ..generate_ast import emit_main_def

    env = CompileEnvironment.current()
    proof = analyze_result_fission(env, func.device_ir, allow_unbound=False)
    if proof is None:
        # The config was admitted on a registration-time proof; the final
        # bound proof must hold, never silently fall back.
        raise exc.BackendUnsupported(
            "cute", "result-fission final bound proof was lost"
        )
    if not all(
        tensor_has_specialized_tma_alignment(env, tensor)
        for tensor in (proof.lhs, proof.rhs, proof.v_tensor)
    ):
        raise exc.BackendUnsupported(
            "cute", "result-fission producer tensors lack TensorMap alignment proof"
        )
    shell = GenerateAST(
        func,
        config,
        codegen_graphs=[_empty_graph(info.graph_id) for info in func.device_ir.graphs],
    )
    launch_marker = ast.Pass()
    roots = 0
    with shell.device_function:
        for statement in func.body:
            if (
                isinstance(statement, ExtendedAST)
                and statement._loop_type is LoopType.GRID
            ):
                shell.add_statement(launch_marker)
                roots += 1
            else:
                shell.add_statement(shell.visit(statement))
    host = func.codegen_function_def(shell.host_statements)
    if roots != 1:
        raise exc.BackendUnsupported(
            "cute", "result fission requires one ordered host launch"
        )
    producer_source, producer_plans, producer_normalized = _producer_source(
        proof, env, config
    )
    consumer_source, consumer_plans, consumer_normalized = _consumer_source(
        proof, env, func, config
    )
    _require_pdl_free("producer", producer_source, producer_plans)
    _require_pdl_free("consumer", consumer_source, consumer_plans)
    tables = [symtable.symtable(ast.unparse(host), "<helion result fission>", "exec")]
    used: set[str] = set()
    while tables:
        table = tables.pop()
        used.update(table.get_identifiers())
        tables.extend(table.get_children())

    def unique(name: str) -> str:
        while name in used:
            name += "_"
        used.add(name)
        return name

    producer_module = unique("_helion_result_fission_producer")
    consumer_module = unique("_helion_result_fission_consumer_module")
    producer_kernel = f"{producer_module}._helion_{PRODUCER_FUNCTION}"
    consumer_kernel = f"{consumer_module}._helion_{CONSUMER_FUNCTION}"
    calls = ast.parse(
        f"{producer_module}.{PRODUCER_FUNCTION}({proof.lhs_expression}, {proof.rhs_expression}, {proof.v_output}, _launcher=_launcher)\n"
        f"{consumer_module}.{CONSUMER_FUNCTION}({', '.join(proof.consumer_params)}, _launcher=_launcher)\n"
    ).body
    index = host.body.index(launch_marker)
    host.body[index : index + 1] = calls
    loads = [statement_from_string("from torch._inductor.codecache import PyCodeCache")]
    for module_name, kernel, source in (
        (producer_module, producer_kernel, producer_source),
        (consumer_module, consumer_kernel, consumer_source),
    ):
        loads.extend(
            (
                statement_from_string(f"{module_name} = PyCodeCache.load({source!r})"),
                statement_from_string(
                    f"{kernel}._helion_cute_source_hash = {hashlib.sha256(source.encode()).hexdigest()!r}"
                ),
            )
        )
    metadata = {
        "identity_output": proof.v_output,
        "outputs": list(proof.outputs),
        "producer": producer_normalized,
        "consumer": consumer_normalized,
        "pdl": False,
        "consumer_statements": list(proof.consumer_statements),
    }
    result = ast.Module(
        body=[
            *func.codegen_imports(),
            *loads,
            host,
            statement_from_string(
                f"{func.name}._helion_cute_kernels = ({producer_kernel}, {consumer_kernel})"
            ),
            statement_from_string(
                f"{func.name}._helion_cute_result_fission = {metadata!r}"
            ),
            *(
                [func.codegen_call_function(), emit_main_def()]
                if emit_repro_caller
                else []
            ),
        ],
        type_ignores=[],
    )
    imports = {
        ast.unparse(stmt)
        for stmt in result.body
        if isinstance(stmt, (ast.Import, ast.ImportFrom))
    }
    result.body[:0] = [
        statement_from_string(line)
        for line in get_needed_import_lines(result)
        if line not in imports
    ]
    env.cute_resolved_wrapper_plans = [*producer_plans, *consumer_plans]
    return result
