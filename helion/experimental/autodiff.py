from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import pathlib
import tempfile
from typing import TYPE_CHECKING

from torch._functorch.aot_autograd import aot_module_simplified
import torch._functorch.config as functorch_config
from torch._functorch.partitioners import min_cut_rematerialization_partition
from torch._inductor.decomposition import select_decomp_table
import torch.fx
from torch.fx.node import Node
from torch.fx.node import map_arg

from .. import exc

if TYPE_CHECKING:
    from ..runtime.kernel import Kernel


@dataclass
class InputMapping:
    placeholder_name: str
    tensor_name: str
    fake_tensor: torch.Tensor | None


class GraphAnalyzer:
    """
    Analyzes forward Helion graph and extracts the pure computation subgraph.
    """

    def __init__(self, forward_graph: torch.fx.Graph) -> None:
        self.forward_graph = forward_graph

    def _get_tensor_name(self, host_tensor_node: Node) -> str:
        target = host_tensor_node.target
        assert callable(target) and getattr(target, "__name__", "") == "_host_tensor"
        name = host_tensor_node.args[0]
        assert isinstance(name, str)
        return name

    def extract_computation_graph(
        self,
    ) -> tuple[torch.fx.Graph, list[InputMapping]]:
        """
        Extract computation subgraph.

        Returns:
            compute_graph: Pure PyTorch FX graph
            input_mappings: Load -> placeholder mappings
        """
        compute_graph = torch.fx.Graph()
        node_map: dict[Node, Node] = {}
        input_mappings: list[InputMapping] = []

        # Track current value of each tensor (None = need placeholder for original)
        tensor_to_placeholder: dict[str, Node] = {}
        tensor_current_value: dict[str, Node] = {}

        # Process nodes in order to preserve load/store semantics
        for node in self.forward_graph.nodes:
            if node.op != "call_function":
                continue

            target = node.target
            assert callable(target)
            target_name = target.__name__

            if target_name == "load":
                host_tensor_node = node.args[0]
                assert isinstance(host_tensor_node, Node)
                tensor_name = self._get_tensor_name(host_tensor_node)
                fake_tensor = node.meta["val"]

                # Check if there's a prior store to this tensor
                if tensor_name in tensor_current_value:
                    # Load after store: use the stored value
                    stored_value_node = tensor_current_value[tensor_name]
                    node_map[node] = node_map[stored_value_node]
                elif tensor_name in tensor_to_placeholder:
                    # Another load of same tensor (before any store): reuse placeholder
                    node_map[node] = tensor_to_placeholder[tensor_name]
                else:
                    # First load of this tensor: create placeholder
                    ph_name = f"tile_{tensor_name}"
                    ph = compute_graph.placeholder(ph_name)
                    tensor_to_placeholder[tensor_name] = ph
                    node_map[node] = ph

                    input_mappings.append(
                        InputMapping(
                            placeholder_name=ph_name,
                            tensor_name=tensor_name,
                            fake_tensor=fake_tensor,
                        )
                    )

            elif target_name == "store":
                host_tensor_node = node.args[0]
                assert isinstance(host_tensor_node, Node)
                tensor_name = self._get_tensor_name(host_tensor_node)
                value_node = node.args[2]
                if isinstance(value_node, Node):
                    tensor_current_value[tensor_name] = value_node

            elif target_name not in ("_host_tensor", "_get_symnode"):
                # Computation node: copy to computation graph
                args = list(node.args)
                # Helion's strip_unused_inputs replaces duplicate node args with None when they map to the same input
                # buffer (e.g., val * val -> mul(val, None)). We restore the real arg for differentiate_graph
                first_node_arg = next((a for a in args if isinstance(a, Node)), None)
                if first_node_arg is not None:
                    args = [first_node_arg if a is None else a for a in args]

                new_args = map_arg(tuple(args), node_map.get)
                new_kwargs = map_arg(node.kwargs, node_map.get)
                target = node.target
                assert callable(target)
                new_node = compute_graph.call_function(target, new_args, new_kwargs)
                if node.meta:
                    new_node.meta = node.meta.copy()
                node_map[node] = new_node

        input_tensor_names = set(tensor_to_placeholder.keys())
        outputs = [
            node_map[v]
            for t, v in tensor_current_value.items()
            if t not in input_tensor_names
        ]
        compute_graph.output(tuple(outputs))
        return compute_graph, input_mappings


def differentiate_graph(
    compute_graph: torch.fx.Graph,
    input_tensors: tuple[torch.Tensor, ...],
) -> torch.fx.Graph:
    """
    Differentiate computation graph using AOT Autograd with full recomputation.

    Returns:
        backward_graph: FX graph for backward computation
    """
    example_inputs = [
        torch.empty(t.shape, dtype=t.dtype, device=t.device, requires_grad=True)
        for t in input_tensors
    ]

    # Capture backward graph via compiler callback
    backward_graph: torch.fx.Graph | None = None

    def bw_compiler(
        gm: torch.fx.GraphModule,
        example_inputs: list[torch.Tensor],
    ) -> torch.fx.GraphModule:
        nonlocal backward_graph
        backward_graph = gm.graph
        return gm

    with functorch_config.patch(activation_memory_budget=0):
        compiled = aot_module_simplified(
            torch.fx.GraphModule({}, compute_graph),
            example_inputs,
            fw_compiler=lambda gm, _: gm,  # type: ignore[arg-type]
            bw_compiler=bw_compiler,  # type: ignore[arg-type]
            decompositions=select_decomp_table(),
            partition_fn=min_cut_rematerialization_partition,
        )

        example_out = compiled(*example_inputs)
        if isinstance(example_out, (list, tuple)):
            loss = sum(o.sum() for o in example_out)
        else:
            loss = example_out.sum()
        assert isinstance(loss, torch.Tensor)
        loss.backward()

    assert backward_graph is not None
    return backward_graph


class FXToHelionConverter:
    """Converts backward FX graph to Helion kernel source code."""

    def __init__(
        self,
        backward_graph: torch.fx.Graph,
        input_mappings: list[InputMapping],
        input_tensors: tuple[torch.Tensor, ...],
    ) -> None:
        self.backward_graph = backward_graph
        self.grad_input_order = [m.tensor_name for m in input_mappings]

        # Map primal index (1-based from AOT Autograd) to tensor name
        self.primal_to_name = {
            i + 1: m.tensor_name for i, m in enumerate(input_mappings)
        }

        # Map tensor name to concrete shape from real input tensors
        self.tensor_shapes: dict[str, tuple[int, ...]] = {
            m.tensor_name: tuple(input_tensors[i].shape)
            for i, m in enumerate(input_mappings)
        }

    def convert(self) -> str:
        """Convert the backward FX graph to Helion kernel source code."""
        placeholders = []
        computations = []
        output_node = None

        for node in self.backward_graph.nodes:
            if node.op == "placeholder":
                placeholders.append(node)
            elif node.op == "call_function":
                computations.append(node)
            elif node.op == "output":
                output_node = node

        # Generate code components
        input_params = ["grad_out", *self.grad_input_order]
        output_grad_names = [f"grad_{name}" for name in self.grad_input_order]
        computation_lines = self._generate_computation(computations, placeholders)
        output_assignments = self._generate_output_assignments(output_node)

        return self._build_source(
            input_params, output_grad_names, computation_lines, output_assignments
        )

    def _get_var_name(self, node_name: str) -> str:
        """Map backward graph node name to generated variable name."""
        if node_name.startswith("primals_"):
            idx = int(node_name.split("_")[1])
            return f"{self.primal_to_name[idx]}_tile"
        if node_name.startswith("tangents_"):
            return "grad_out_tile"
        return f"{node_name}_val"

    def _generate_computation(
        self, computations: list[Node], placeholders: list[Node]
    ) -> list[str]:
        """Generate Python code for each computation node.

        With full recomputation (activation_memory_budget=0), the backward graph
        contains all forward computation ops. Placeholders are only primals_
        (original inputs) and tangents_ (upstream gradients).
        """
        lines = []
        node_to_var: dict[str, str] = {}

        def process_arg(arg: object) -> str:
            if isinstance(arg, Node):
                return node_to_var[arg.name]
            if isinstance(arg, (list, tuple)):
                processed = [process_arg(item) for item in arg]
                return f"[{', '.join(processed)}]"
            return repr(arg)

        # Map all placeholders (primals_ and tangents_) to variable names
        for ph in placeholders:
            node_to_var[ph.name] = self._get_var_name(ph.name)

        # Generate code for each computation node
        for node in computations:
            target = node.target
            op_name = getattr(target, "_opname", None)

            # Skip identity ops - just alias to input variable
            if op_name in {"detach", "alias"}:
                if node.args:
                    input_node = node.args[0]
                    assert isinstance(input_node, Node)
                    node_to_var[node.name] = node_to_var[input_node.name]
                    continue

            # Skip scalar_tensor - will inline the literal value
            if op_name == "scalar_tensor":
                if node.args:
                    node_to_var[node.name] = repr(node.args[0])
                    continue

            result_var = f"{node.name}_val"
            node_to_var[node.name] = result_var

            arg_vars = [process_arg(arg) for arg in node.args]

            # Generate op code: torch function or tensor method
            if op_name is not None and arg_vars:
                if hasattr(torch, op_name):
                    code = f"torch.{op_name}({', '.join(arg_vars)})"
                else:
                    tensor = arg_vars[0]
                    method_args = ", ".join(arg_vars[1:])
                    code = f"{tensor}.{op_name}({method_args})"
            elif op_name is not None:
                code = f"torch.{op_name}({', '.join(arg_vars)})"
            else:
                code = f"{node.target}({', '.join(arg_vars)})"

            lines.append(f"{result_var} = {code}")

        return lines

    def _generate_output_assignments(
        self, output_node: Node | None
    ) -> list[tuple[str, str]]:
        """
        Map backward graph outputs to gradient variable assignments.
        The backward graph returns gradients in the same order as forward inputs
        """
        if output_node is None:
            return []

        # FX output node stores return values in args[0]
        output_args = output_node.args[0]

        if not isinstance(output_args, (list, tuple)):
            output_args = [output_args]

        # Pair each output with its corresponding gradient name
        assignments = []
        assert isinstance(output_args, (list, tuple))
        for i, out_node in enumerate(output_args):
            grad_name = f"grad_{self.grad_input_order[i]}"
            assert isinstance(out_node, Node)
            var_name = self._get_var_name(out_node.name)
            assignments.append((grad_name, var_name))

        return assignments

    def _build_source(
        self,
        input_params: list[str],
        output_grad_names: list[str],
        computation_lines: list[str],
        output_assignments: list[tuple[str, str]],
    ) -> str:
        """Build the complete Helion kernel source code using AST."""

        # Iteration shape comes from the first output gradient's tensor
        iter_var = output_grad_names[0]
        iter_tensor_name = iter_var.replace("grad_", "")
        iter_ndim = len(self.tensor_shapes[iter_tensor_name])

        def parse_expr(code: str) -> ast.expr:
            return ast.parse(code, mode="eval").body

        def parse_stmt(code: str) -> ast.stmt:
            return ast.parse(code, mode="exec").body[0]

        # Build imports
        imports: list[ast.stmt] = [
            ast.Import(names=[ast.alias(name="torch", asname=None)]),
            ast.Import(names=[ast.alias(name="helion", asname=None)]),
            ast.ImportFrom(
                module="helion",
                names=[ast.alias(name="language", asname="hl")],
                level=0,
            ),
        ]

        # Build function parameters: (grad_out: torch.Tensor, x: torch.Tensor, ...)
        tensor_annotation = parse_expr("torch.Tensor")
        func_args = ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=p, annotation=tensor_annotation) for p in input_params],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        )

        # Return type: torch.Tensor or tuple[torch.Tensor, ...]
        multi_output = len(output_grad_names) > 1
        return_annotation = parse_expr(
            "tuple[torch.Tensor, ...]" if multi_output else "torch.Tensor"
        )

        # Function body before loop: grad_x = torch.empty_like(x)
        body: list[ast.stmt] = [
            parse_stmt(f"{g} = torch.empty_like({g.replace('grad_', '')})")
            for g in output_grad_names
        ]

        # Loop body: loads → computation → stores
        loop_body: list[ast.stmt] = []

        # Load statements
        for p in input_params:
            tensor_ndim = iter_ndim if p == "grad_out" else len(self.tensor_shapes[p])
            if tensor_ndim < iter_ndim:
                indices = ", ".join(f"tile[{i}]" for i in range(tensor_ndim))
                load_expr = f"{p}[{indices}]"
            else:
                load_expr = f"{p}[tile]"
            loop_body.append(parse_stmt(f"{p}_tile = {load_expr}"))

        # Computation statements
        for line in computation_lines:
            loop_body.append(parse_stmt(line))

        # Store statements: grad_x[tile] = value
        for grad_name, var_name in output_assignments:
            loop_body.append(parse_stmt(f"{grad_name}[tile] = {var_name}"))

        # For loop: for tile in hl.tile(iter_var.shape):
        body.append(
            ast.For(
                target=ast.Name(id="tile", ctx=ast.Store()),
                iter=parse_expr(f"hl.tile({iter_var}.shape)"),
                body=loop_body,
                orelse=[],
            )
        )

        # Return statement
        if multi_output:
            return_value = ast.Tuple(
                elts=[ast.Name(id=g, ctx=ast.Load()) for g in output_grad_names],
                ctx=ast.Load(),
            )
        else:
            return_value = ast.Name(id=output_grad_names[0], ctx=ast.Load())
        body.append(ast.Return(value=return_value))

        # Function definition with @helion.kernel() decorator
        func_def = ast.FunctionDef(
            name="backward_kernel",
            args=func_args,
            body=body,
            decorator_list=[parse_expr("helion.kernel()")],
            returns=return_annotation,
        )

        module = ast.Module(body=[*imports, func_def], type_ignores=[])
        ast.fix_missing_locations(module)

        source = ast.unparse(module)
        header = '"""\nAuto-generated Helion backward kernel.\n"""\n\n'
        return header + source


def backward(
    kernel: Kernel[object],
    grad_out: torch.Tensor,
    *inputs: torch.Tensor,
    return_code: bool = False,
    autotune: bool = False,
    autotune_effort: str | None = None,
) -> (
    tuple[torch.Tensor, ...]
    | torch.Tensor
    | tuple[tuple[torch.Tensor, ...] | torch.Tensor, str, str]
):
    """
    Compute gradients for a Helion kernel.

    The backward kernel is generated as an independent Helion kernel with its
    own ConfigSpec, allowing separate autotuning from the forward kernel.

    Args:
        kernel: A @helion.kernel decorated function (must be called once first)
        grad_out: Gradient of loss w.r.t. kernel output
        *inputs: The original inputs to the kernel (in the same order as forward)
        return_code: If True, also return the generated backward kernel code
        autotune: If True, autotune the backward kernel for best performance
        autotune_effort: Autotuning effort level ('none', 'quick', 'full').
            Default is 'none' when autotune=False, 'quick' when autotune=True.

    Returns:
        If return_code=False: Tuple of gradients (or single tensor if one input)
        If return_code=True: (gradients, helion_code, triton_code) tuple

    Example:
        @helion.kernel()
        def my_kernel(x, y):
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                out[tile] = x[tile] * y[tile]
            return out

        out = my_kernel(x, y)
        grad_x, grad_y = helion.experimental.backward(my_kernel, grad_out, x, y)
    """
    if not hasattr(kernel, "_bound_kernels") or not kernel._bound_kernels:
        raise exc.AutodiffKernelNotCalled

    bound = kernel.bind(inputs)
    if bound._config is None:
        bound._config = bound.env.config_spec.default_config()

    if bound._backward_compiled is not None:
        bwd_fn, bwd_source, bwd_bound = bound._backward_compiled
    else:
        from .._compiler.device_ir import ForLoopGraphInfo
        from .._compiler.device_ir import ReductionLoopGraphInfo
        from .._compiler.device_ir import RootGraphInfo

        host_function = bound.host_function
        assert host_function is not None
        graphs = host_function.device_ir.graphs

        # Only support single RootGraphInfo (simple elementwise kernels)
        if len(graphs) != 1 or not isinstance(graphs[0], RootGraphInfo):
            for graph_info in graphs:
                if isinstance(graph_info, ReductionLoopGraphInfo):
                    raise exc.AutodiffNotSupported("reduction operations")
                if isinstance(graph_info, ForLoopGraphInfo):
                    raise exc.AutodiffNotSupported("multiple tile loops")
            raise exc.AutodiffNotSupported("multiple graphs")

        fwd_graph = graphs[0].graph

        analyzer = GraphAnalyzer(fwd_graph)
        compute_graph, input_mappings = analyzer.extract_computation_graph()

        backward_graph = differentiate_graph(compute_graph, inputs)

        converter = FXToHelionConverter(
            backward_graph=backward_graph,
            input_mappings=input_mappings,
            input_tensors=inputs,
        )
        bwd_source = converter.convert()

        with tempfile.TemporaryDirectory(prefix="helion_bwd_") as cache_dir:
            source_hash = hashlib.md5(
                bwd_source.encode(), usedforsecurity=False
            ).hexdigest()[:12]
            temp_path = pathlib.Path(cache_dir) / f"helion_bwd_{source_hash}.py"

            temp_path.write_text(bwd_source)

            spec = importlib.util.spec_from_file_location(
                f"helion_bwd_{source_hash}", str(temp_path)
            )
            assert spec is not None and spec.loader is not None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            assert hasattr(module, "backward_kernel")

            bwd_fn = module.backward_kernel
            bwd_args = (grad_out, *inputs)
            bwd_bound = bwd_fn.bind(bwd_args)

            # Determine autotune_effort: use 'quick' when autotuning, 'none' otherwise
            if autotune_effort is None:
                autotune_effort = "quick" if autotune else "none"

            # Set autotune_effort to prevent automatic autotuning on first call
            bwd_bound.settings.autotune_effort = autotune_effort
            if autotune:
                bwd_bound.autotune(bwd_args)
            bound._backward_compiled = (bwd_fn, bwd_source, bwd_bound)

    result = bwd_fn(grad_out, *inputs)
    if isinstance(result, tuple):
        assert all(isinstance(r, torch.Tensor) for r in result)
        grads: torch.Tensor | tuple[torch.Tensor, ...] = (
            result if len(result) > 1 else result[0]
        )
    else:
        assert isinstance(result, torch.Tensor)
        grads = result

    if return_code:
        if bwd_bound._config is None:
            bwd_bound._config = bwd_bound.env.config_spec.default_config()
        triton_code: str = bwd_bound.to_triton_code(bwd_bound._config)
        return grads, bwd_source, triton_code

    return grads
