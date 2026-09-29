"""Generate an optional guarded cache variant beside its original kernel.

Both variants use independent DeviceFunction instances and the ordinary lowering
pipeline. Admission never changes the caller's configuration, input schema, or
fallback module. The final bundle lists and hashes both kernels for existing
CuTe precompile, serialization, and cache-lifetime handling.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from dataclasses import field
import hashlib
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch

from ...runtime.config import Config
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..host_function import HostFunction
from .bounded_loop_cache import BoundedLoopPlan
from .bounded_loop_cache import OwnedFragment
from .bounded_loop_cache import attach_variant
from .bounded_loop_cache import cache_single_tile_loads
from .bounded_loop_cache import specialize_single_tile_sweeps
from .duplicate_tile_maxima import deduplicate_bounded_maxima
from .tensor_layout_relations import TensorLayoutRelation
from .tensor_layout_relations import prove_tensor_layout_relations

if TYPE_CHECKING:
    from collections.abc import Collection
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ..device_function import Argument
    from ..device_function import DeviceFunction


BOUNDED_SEQUENCE_MODES = frozenset({"bounded", "bounded_layout"})


def _launch_block(df: DeviceFunction) -> tuple[int, int, int] | None:
    """Read the emitted launcher; strategy x/y order is not authoritative."""
    calls = [
        node
        for statement in df.codegen.host_statements
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_launcher"
    ]
    if len(calls) != 1:
        return None
    call = calls[0]
    if (
        not call.args
        or not isinstance(call.args[0], ast.Name)
        or call.args[0].id != df.name
    ):
        return None
    block = next((kw.value for kw in call.keywords if kw.arg == "block"), None)
    if (
        not isinstance(block, ast.Tuple)
        or len(block.elts) != 3
        or any(
            not isinstance(item, ast.Constant) or type(item.value) is not int
            for item in block.elts
        )
    ):
        return None
    return cast(
        "tuple[int, int, int]",
        tuple(cast("ast.Constant", item).value for item in block.elts),
    )


@dataclass
class BoundedCacheRequest:
    use_layout_relations: bool
    plan: BoundedLoopPlan | None = None
    relations: tuple[TensorLayoutRelation, ...] = ()
    argument_names: tuple[str, ...] = ()
    tensor_dtypes: dict[str, str] = field(default_factory=dict)
    disjoint_pairs: Collection[frozenset[str]] = ()
    completed: bool = False

    def prepare(
        self,
        body: list[ast.stmt],
        df: DeviceFunction,
        arguments: Sequence[Argument],
        constants: Mapping[str, int],
        tensor_dtypes: Mapping[str, str],
        disjoint_pairs: Collection[frozenset[str]],
    ) -> list[ast.stmt]:
        from ..device_function import NumericArgument
        from ..device_function import TensorArg
        from ..device_function import TensorDescriptorArg
        from ..device_function import TensorPropertyArg

        if (
            df.codegen.cute_wrapper_plans
            or df.codegen.cute_uses_matmul
            or df.cute_state.simt_cluster_n != 1
            or df.has_rng_ops()
            or df._scratch_args
            or df.wrapper_only_params
            or df.codegen._extra_params
            or any(isinstance(arg, TensorDescriptorArg) for arg in arguments)
        ):
            return body
        launch_block = _launch_block(df)
        if launch_block is None:
            return body
        integer_names = {
            argument.name
            for expression, argument in df._expr_args.items()
            if isinstance(expression, sympy.Symbol)
            and expression.assumptions0.get("integer") is True
        } | {
            argument.name
            for argument in arguments
            if isinstance(argument, TensorPropertyArg)
        }
        # NumericArgument includes floating scalars; only the exact integer
        # subset above is allowed to govern the new host guard or relation.
        if any(not isinstance(arg, (TensorArg, NumericArgument)) for arg in arguments):
            return body
        prepared = specialize_single_tile_sweeps(
            body,
            integer_arguments=integer_names,
            constexpr_values=constants,
            launch_block=launch_block,
        )
        if prepared is None:
            return body
        self.plan = prepared.plan
        self.argument_names = tuple(arg.name for arg in arguments)
        self.tensor_dtypes = dict(tensor_dtypes)
        self.disjoint_pairs = disjoint_pairs
        if self.use_layout_relations:
            inputs = HostFunction.current().params.arguments
            self.relations = prove_tensor_layout_relations(
                cast("list[ast.stmt]", df.codegen.host_statements),
                kernel_name=df.name,
                parameter_names=self.argument_names,
                tensor_parameters={
                    arg.name: arg.fake_value.ndim
                    for arg in arguments
                    if isinstance(arg, TensorArg)
                },
                integer_parameters=integer_names,
                tensor_inputs={
                    name: value.ndim
                    for name, value in inputs.items()
                    if isinstance(value, torch.Tensor)
                },
                scalar_inputs={
                    name
                    for name, value in inputs.items()
                    if isinstance(
                        value, (bool, int, float, torch.SymInt, torch.SymFloat)
                    )
                },
            )
        return prepared.body

    def cache(
        self,
        body: list[ast.stmt],
        constants: Mapping[str, int],
        rename_groups: Mapping[str, str],
    ) -> tuple[list[ast.stmt], tuple[OwnedFragment, ...]]:
        if self.plan is None:
            return body, ()
        cached = cache_single_tile_loads(
            body,
            self.plan,
            argument_names=self.argument_names,
            constexpr_values=constants,
            tensor_dtypes=self.tensor_dtypes,
            proven_disjoint_tensor_pairs=self.disjoint_pairs,
            rename_groups=rename_groups,
        )
        if cached is None:
            return body, ()
        self.completed = True
        return (
            deduplicate_bounded_maxima(
                cached,
                argument_names=self.argument_names,
                constexpr_values=constants,
                rename_groups=rename_groups,
            ),
            cached.fragments,
        )

    def finalize(self, body: list[ast.stmt]) -> list[ast.stmt]:
        # Insert relations after all address/alias/vector proofs. They are
        # exact runtime identities, but those existing proof vocabularies
        # need not learn cute.size or a new write to a parameter name.
        if not self.completed:
            return body
        return [*(relation.assignment() for relation in self.relations), *body]


def generate_bounded_cache(
    host: HostFunction, config: Config, emit_repro_caller: bool
) -> ast.Module:
    from ..generate_ast import generate_ast

    # Every ordinary path is generated first and remains the exact fallback.
    # A failed proof returns that original AST without changing its signature,
    # launch, annotations, or serialization contract.
    scalar = Config.from_dict({**config.config, "cute_reduction_sequence": "scalar"})
    original = generate_ast(host, scalar, emit_repro_caller)
    env = CompileEnvironment.current()
    if (
        env.cute_resolved_wrapper_plans
        or env.cute_fission_plan is not None
        or env.config_spec.matmul_facts
        or len(host.device_ir.root_ids) != 1
        or len(host.device_ir.phases) != 1
    ):
        return original
    request = BoundedCacheRequest(
        use_layout_relations=config.get("cute_reduction_sequence") == "bounded_layout"
    )
    fast = generate_ast(host, scalar, emit_repro_caller, _bounded_cache_request=request)
    if not request.completed or request.plan is None:
        return original
    kernel_name = f"_helion_{host.name}"
    # The initial envelope is ordinary kernels with identical module/host
    # bindings. The checked attachment declines metadata, tensor maps, scratch
    # parameters and other launch wrappers rather than synthesizing a schema.
    source = attach_variant(
        ast.unparse(ast.fix_missing_locations(original)),
        ast.unparse(ast.fix_missing_locations(fast)),
        request.plan,
        kernel_name=kernel_name,
    )
    if source is None:
        return original
    result = ast.parse(source)
    kernels = [
        node.name
        for node in result.body
        if isinstance(node, ast.FunctionDef)
        and any(ast.unparse(item) == "cute.kernel" for item in node.decorator_list)
    ]
    assert len(kernels) == 2 and kernel_name in kernels
    for name in kernels:
        # Include both bodies and all imports/constants, and distinguish the
        # conditional kernels even when they have identical input schemas.
        digest = hashlib.sha256((source + "\n" + name).encode()).hexdigest()
        result.body.append(
            statement_from_string(f"{name}._helion_cute_source_hash = {digest!r}")
        )
    result.body.append(
        statement_from_string(
            f"{host.name}._helion_cute_kernels = ({', '.join(kernels)})"
        )
    )
    return result
