from __future__ import annotations

from abc import ABC
from abc import abstractmethod
import ast
import inspect
from typing import TYPE_CHECKING
from typing import Callable
from typing import Literal
from typing import cast

import torch

from .ast_extension import create
from .ast_extension import create_arg
from .ast_extension import create_arguments
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string

if TYPE_CHECKING:
    import types

    from .device_function import DeviceFunction
    from .device_ir import HelperFunctionGraphInfo


class CodegenInterface(ABC):
    """Abstract base class for codegen interfaces used by GraphInterpreter."""

    def __init__(self, device_function: DeviceFunction) -> None:
        self.device_function = device_function

    @abstractmethod
    def add_statement(self, stmt: ast.AST | str | None) -> None:
        """Add a statement to the generated code."""

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


def extract_helper_function(helper_fn: object) -> types.FunctionType:
    """Extract the actual function from a Kernel object or return as-is.

    This utility function centralizes the logic for handling both regular functions
    and Kernel objects that wrap functions.
    """
    from ..runtime.kernel import Kernel

    return helper_fn.fn if isinstance(helper_fn, Kernel) else helper_fn  # pyright: ignore[reportReturnType]


def extract_helper_function_name(helper_fn: object) -> str:
    """Extract the function name from a Kernel object or regular function."""
    return extract_helper_function(helper_fn).__name__


CombineFunctionBasic = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
CombineFunctionTuple = Callable[..., tuple[torch.Tensor, ...]]
CombineFunction = CombineFunctionBasic | CombineFunctionTuple


def create_combine_function_wrapper(
    combine_fn: CombineFunction,
    *,
    is_tuple_input: bool,
    target_format: Literal["tuple", "unpacked"],
) -> CombineFunction:
    """
    Create a wrapper around combine_fn that converts between different combine function formats.

    Args:
        combine_fn: The original combine function
        is_tuple_input: Whether the input is a tuple
        target_format: Either 'tuple' or 'unpacked'. The 'tuple' option expects
            (left_tuple, right_tuple) inputs, while 'unpacked' expects
            (left_elem0, left_elem1, ..., right_elem0, right_elem1, ...) inputs

    Returns:
        A wrapper function that converts between the formats
    """
    # Extract the actual function (handles @helion.kernel decorated functions)
    actual_fn = extract_helper_function(combine_fn)

    # For single tensor inputs, no conversion needed
    if not is_tuple_input:
        return actual_fn

    # Inspect the original function signature to determine its format
    sig = inspect.signature(actual_fn)
    param_count = len(sig.parameters)

    # Determine the original format based on parameter count
    # If it has 2 parameters, it's tuple format: (left_tuple, right_tuple)
    # If it has 4+ parameters, it's unpacked format: (left_elem0, left_elem1, ..., right_elem0, right_elem1, ...)
    original_format = "tuple" if param_count < 4 else "unpacked"

    # If the original format matches target format, no conversion needed
    if target_format == original_format:
        return actual_fn
    combine_fn = cast("CombineFunctionTuple", combine_fn)

    # Create conversion wrapper
    if target_format == "tuple" and original_format == "unpacked":
        # Convert from unpacked to tuple format
        # Target: (left_tuple, right_tuple)
        # Original: (left_elem0, left_elem1, ..., right_elem0, right_elem1, ...)
        def tuple_wrapper(
            left_tuple: tuple[torch.Tensor, ...], right_tuple: tuple[torch.Tensor, ...]
        ) -> tuple[torch.Tensor, ...]:
            return combine_fn(*left_tuple, *right_tuple)

        return tuple_wrapper

    if target_format == "unpacked" and original_format == "tuple":
        # Convert from tuple to unpacked format
        # Target: (left_elem0, left_elem1, ..., right_elem0, right_elem1, ...)
        # Original: (left_tuple, right_tuple)
        def unpacked_wrapper(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
            num_args = len(args)
            assert (num_args % 2) == 0
            half = num_args // 2
            left_tuple = args[:half]
            right_tuple = args[half:]
            return combine_fn((*left_tuple,), (*right_tuple,))

        return unpacked_wrapper

    # Should not reach here
    raise ValueError(
        f"Unsupported conversion from {original_format} to {target_format}"
    )


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

    def register_helper_function(
        self, helper_graph_info: HelperFunctionGraphInfo, final_name: str
    ) -> None:
        """Register a helper function to be generated at global scope."""
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
                decorator_list=[expr_from_string("triton.jit")],
                type_params=[],
            )

            helper_defs.append(func_def)

        return helper_defs

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
