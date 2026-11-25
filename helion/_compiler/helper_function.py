from __future__ import annotations

from abc import ABC
from abc import abstractmethod
import ast
import contextlib
import inspect
from typing import TYPE_CHECKING
from typing import Callable
from typing import Literal
from typing import cast

import torch
from torch.utils._pytree import tree_map_only

from .ast_extension import create
from .ast_extension import create_arg
from .ast_extension import create_arguments
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment

if TYPE_CHECKING:
    from collections.abc import Iterator
    import types

    from torch.fx.node import Node

    from .device_function import DeviceFunction
    from .device_ir import HelperFunctionGraphInfo


class CodegenInterface(ABC):
    """Abstract base class for codegen interfaces used by GraphInterpreter."""

    def __init__(self, device_function: DeviceFunction) -> None:
        self.device_function = device_function

    @abstractmethod
    def add_statement(self, stmt: ast.AST | str | None) -> None:
        """Add a statement to the generated code."""

    @contextlib.contextmanager
    def statement_owner_node(self, node: Node) -> Iterator[None]:
        yield

    def tmpvar(self, *, dce: bool = False, prefix: str = "v") -> str:
        """Generate a temporary variable name."""
        return self.device_function.unique_name(prefix, dce=dce)

    def lift(self, expr: ast.AST, *, dce: bool = False, prefix: str = "v") -> ast.Name:
        """Lift an expression to a temporary variable if needed."""
        if isinstance(expr, ast.Name):
            return expr
        varname = self.tmpvar(dce=dce, prefix=prefix)
        self.add_statement(statement_from_string(f"{varname} = {{expr}}", expr=expr))
        return create(ast.Name, id=varname, ctx=ast.Load())

    def rolled_reduction_index_var(self, size: object) -> str | None:
        """Index of the rolled reduction loop over the dim of extent ``size``."""
        return None


def extract_helper_function(helper_fn: object) -> types.FunctionType:
    """Extract the actual function from a Kernel object or return as-is.

    This utility function centralizes the logic for handling both regular functions
    and Kernel objects that wrap functions.
    """
    from ..runtime.kernel import Kernel

    # pyrefly: ignore [bad-return]
    return helper_fn.fn if isinstance(helper_fn, Kernel) else helper_fn


def extract_helper_function_name(helper_fn: object) -> str:
    """Extract the function name from a Kernel object or regular function."""
    return extract_helper_function(helper_fn).__name__


CombineFunctionBasic = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
CombineFunctionTuple = Callable[..., tuple[torch.Tensor, ...]]
CombineFunction = CombineFunctionBasic | CombineFunctionTuple


def create_combine_function_wrapper(
    combine_fn: CombineFunction,
    operands: torch.Tensor | tuple[torch.Tensor, ...],
    *,
    target_format: Literal["tuple", "unpacked"],
    reduction_mode: Literal["compiled", "eager"] | None = None,
) -> CombineFunction:
    """Adapt a Python combine to the argument layout used by its caller."""
    actual_fn = extract_helper_function(combine_fn)
    if not isinstance(operands, (tuple, list)):
        # FX gets exactly the two operands, including for a callback with
        # keyword-only defaults or **kwargs.
        return lambda left, right: actual_fn(left, right)

    size = len(operands)
    sig = inspect.signature(actual_fn)
    # Previously, converting a decorated reduction called the Kernel again
    # and failed. Use its working eager layout on the newly compilable path.
    # Plain compiled callbacks and scans retain their existing convention.
    prefer_eager = reduction_mode == "eager" or (
        reduction_mode == "compiled" and actual_fn is not combine_fn
    )
    count = len(sig.parameters)
    flat = count == 2 * size if prefer_eager and size > 1 else count >= 4
    packed_args = (None, None)
    flat_args = packed_args * size
    try:
        sig.bind(*(flat_args if flat else packed_args))
    except TypeError:
        sig.bind(*(packed_args if flat else flat_args))
        flat = not flat

    # Give FX only the actual operands. Passing the original signature would
    # also create placeholders for default parameters Python should supply.
    fn = cast("CombineFunctionTuple", actual_fn)
    if target_format == "tuple":

        def from_tuples(
            left: tuple[torch.Tensor, ...], right: tuple[torch.Tensor, ...]
        ) -> tuple[torch.Tensor, ...]:
            return fn(*left, *right) if flat else fn(left, right)

        return from_tuples

    def from_flat(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return fn(*args) if flat else fn(args[:size], args[size:])

    return from_flat


class HelperCodegen(CodegenInterface):
    """Codegen wrapper for helper function generation."""

    def __init__(self, device_function: DeviceFunction) -> None:
        super().__init__(device_function)

    def add_statement(self, stmt: ast.AST | str | None) -> None:
        if stmt is not None:
            if isinstance(stmt, str):
                stmt = statement_from_string(stmt)
            self.device_function.body.append(stmt)


class HelperFunctionManager:
    """Manages helper function registration and code generation."""

    def __init__(self) -> None:
        self.helper_functions: dict[str, HelperFunctionGraphInfo] = {}
        self._final_names: dict[str, str] = {}
        self._by_structure: dict[tuple[object, ...], str] = {}

    def register_helper_function(
        self, helper_graph_info: HelperFunctionGraphInfo, final_name: str
    ) -> None:
        """Register a helper function to be generated at global scope; one
        identical to a registered helper (every reduction's combine function)
        shares its definition."""
        graph = helper_graph_info.graph
        index = {node: i for i, node in enumerate(graph.nodes)}

        def ref(node: torch.fx.Node) -> torch.fx.node.Argument:
            return ("node", index[node])

        def value(node: torch.fx.Node) -> str:
            # The generated body depends on each value's dtype (casts, constants).
            return repr(
                tree_map_only(
                    torch.Tensor,
                    lambda t: (t.dtype, tuple(t.shape)),
                    node.meta.get("val"),
                )
            )

        structure = tuple(
            (
                node.op,
                str(node.target),
                repr(torch.fx.node.map_arg(node.args, ref)),
                repr(torch.fx.node.map_arg(node.kwargs, ref)),
                value(node),
            )
            for node in graph.nodes
        )
        same = self._by_structure.setdefault(structure, helper_graph_info.name)
        if same != helper_graph_info.name:
            self._final_names[helper_graph_info.name] = self._final_names[same]
            return
        self.helper_functions[helper_graph_info.name] = helper_graph_info
        self._final_names[helper_graph_info.name] = final_name

    def codegen_helper_functions(self) -> list[ast.stmt]:
        """Generate helper function definitions at global scope."""
        helper_defs = []
        for helper_graph_info in self.helper_functions.values():
            # Get the final name that was already determined during registration
            final_name = self._final_names[helper_graph_info.name]

            # Determine the number of parameters from the graph
            input_nodes = helper_graph_info.find_input_nodes()

            # Generate argument list with consistent names
            args = []
            param_names = []
            for i in range(len(input_nodes)):
                arg_name = f"param_{i}"
                args.append(create_arg(arg_name))
                param_names.append(arg_name)

            # Store parameter names for use in body generation
            helper_graph_info._param_names = param_names

            # Process the FX graph to generate the correct helper function body
            func_body = self._codegen_helper_function_body(helper_graph_info)

            # Generate the function structure with @triton.jit decorator
            func_def = create(
                ast.FunctionDef,
                name=final_name,
                args=create_arguments(args),
                body=func_body,
                decorator_list=[
                    expr_from_string(
                        CompileEnvironment.current().backend.helper_decorator
                    )
                ],
                type_params=[],
            )

            helper_defs.append(func_def)

        return helper_defs  # pyrefly: ignore[bad-return]

    def get_final_name(self, helper_graph_info: HelperFunctionGraphInfo) -> str:
        """Get the final generated name for a helper function."""
        return self._final_names.get(helper_graph_info.name, helper_graph_info.name)

    def _codegen_helper_function_body(
        self, helper_graph_info: HelperFunctionGraphInfo
    ) -> list[ast.stmt]:
        """Generate the body of a helper function by processing its FX graph."""
        temp_device_function = self._create_temp_device_function(helper_graph_info)
        param_args = self._create_parameter_args(helper_graph_info)

        with temp_device_function:
            results = self._process_helper_graph(
                helper_graph_info, temp_device_function, param_args
            )
            statements = temp_device_function.body.copy()
            self._ensure_return_statement(statements, results, helper_graph_info.name)

        return cast("list[ast.stmt]", statements)

    def _create_temp_device_function(
        self, helper_graph_info: HelperFunctionGraphInfo
    ) -> DeviceFunction:
        """Create a temporary DeviceFunction for helper function generation."""
        # Import here to avoid circular imports
        from .device_function import DeviceFunction

        current = DeviceFunction.current()

        return DeviceFunction(
            name=f"temp_{helper_graph_info.name}",
            config=current.config,
            codegen=current.codegen,
        )

    def _create_parameter_args(
        self, helper_graph_info: HelperFunctionGraphInfo
    ) -> list[ast.AST]:
        """Create parameter AST nodes for the helper function."""
        param_names = helper_graph_info._param_names
        return [expr_from_string(param_name) for param_name in param_names]

    def _process_helper_graph(
        self,
        helper_graph_info: HelperFunctionGraphInfo,
        temp_device_function: DeviceFunction,
        param_args: list[ast.AST],
    ) -> object:
        """Process the graph using the existing interpreter infrastructure."""
        from .inductor_lowering import GraphInterpreter

        helper_codegen = HelperCodegen(temp_device_function)
        interpreter = GraphInterpreter(helper_graph_info.graph, helper_codegen)
        return interpreter.run(*param_args)

    def _ensure_return_statement(
        self, statements: list[ast.AST], results: object, function_name: str
    ) -> None:
        """Ensure the function body has a proper return statement."""
        if statements and isinstance(statements[-1], ast.Return):
            return

        if isinstance(results, ast.AST):
            statements.append(create(ast.Return, value=results))
        elif isinstance(results, (list, tuple)) and all(
            isinstance(r, ast.AST) for r in results
        ):
            tuple_ast = create(ast.Tuple, elts=list(results), ctx=ast.Load())
            statements.append(create(ast.Return, value=tuple_ast))
        else:
            raise RuntimeError(
                f"Helper function {function_name} produced invalid result: {type(results)} {results}"
            )


def codegen_helper_function_graph_info(
    helper_graph_info: HelperFunctionGraphInfo, state: object
) -> list[object]:
    """Generate code for HelperFunctionGraphInfo objects."""
    from .inductor_lowering import CodegenState
    from .inductor_lowering import codegen_call_with_graph

    if not isinstance(state, CodegenState):
        raise TypeError(f"Expected CodegenState, got {type(state)}")

    # For helper functions, we need to inline the function body
    # The helper function takes variable arguments and returns their combination

    # Generate temporary variable names for the helper function arguments
    # Use the graph's input nodes to determine the number of parameters
    input_nodes = helper_graph_info.find_input_nodes()
    args: list[ast.AST] = []

    for i in range(len(input_nodes)):
        var_name = state.codegen.tmpvar(prefix=f"helper_arg_{i}")
        args.append(create(ast.Name, id=var_name, ctx=ast.Load()))

    # Generate the helper function call
    return codegen_call_with_graph(state.codegen, helper_graph_info.graph, args)
