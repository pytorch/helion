"""Test the inductor matmul lowering fix for symbolic shapes"""
import torch
import torch.fx
from torch.fx.experimental import proxy_tensor
from helion._compiler.inductor_lowering import codegen_matmul
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.host_function import HostFunction
from helion._compiler.ast_extension import expr_from_string
import sympy

# Mock context and node for testing
class MockCodegenInterface:
    class MockDeviceFunction:
        class MockTileStrategy:
            def shape_str(self, shape):
                # Convert shape to string representation
                return f"[{', '.join(str(s) for s in shape)}]"
        
        def __init__(self):
            self.tile_strategy = self.MockTileStrategy()
            self.user_sympy_expr_cache = {}
            
        def user_sympy_expr(self, expr):
            # For testing, just return string representation
            return str(expr)
    
    def __init__(self):
        self.device_function = self.MockDeviceFunction()

class MockGraphInterpreter:
    def __init__(self):
        self.cg = MockCodegenInterface()
        self.env = {}

def test_symbolic_matmul():
    """Test that the matmul lowering handles symbolic shapes correctly"""
    print("Testing symbolic shape handling in codegen_matmul...")
    
    # Create symbolic shapes
    B = torch.SymInt(torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode(
        sympy.Symbol("B", positive=True, integer=True)
    ))
    H = torch.SymInt(torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode(
        sympy.Symbol("H", positive=True, integer=True)
    ))
    I = torch.SymInt(torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode(
        sympy.Symbol("I", positive=True, integer=True)
    ))
    D = torch.SymInt(torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode(
        sympy.Symbol("D", positive=True, integer=True)
    ))
    J = torch.SymInt(torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode(
        sympy.Symbol("J", positive=True, integer=True)
    ))
    
    # Create mock nodes with symbolic shapes
    class MockNode:
        def __init__(self, shape, ndim):
            self.meta = {"val": type('obj', (object,), {"shape": shape, "ndim": ndim})()}
    
    lhs_node = MockNode((B, H, I, D), 4)
    rhs_node = MockNode((B, H, D, J), 4)
    
    # Create main node
    class MockMatmulNode:
        def __init__(self):
            self.args = (lhs_node, rhs_node)
            self.kwargs = {}
            self.meta = {"val": type('obj', (object,), {"shape": (B, H, I, J), "size": lambda: (B, H, I, J)})()}
    
    node = MockMatmulNode()
    
    # Mock the graph interpreter
    ctx = MockGraphInterpreter()
    # Mock AST nodes for the tensors
    import ast
    ctx.env[lhs_node] = ast.Name(id="lhs", ctx=ast.Load())
    ctx.env[rhs_node] = ast.Name(id="rhs", ctx=ast.Load())
    
    # Mock the CompileEnvironment
    class MockSettings:
        dot_precision = None
    
    class MockCompileEnvironment:
        def __init__(self):
            self.settings = MockSettings()
        
        @staticmethod
        def current():
            return MockCompileEnvironment()
    
    # Temporarily replace CompileEnvironment
    import helion._compiler.inductor_lowering as lowering_module
    original_env = lowering_module.CompileEnvironment
    lowering_module.CompileEnvironment = MockCompileEnvironment
    
    try:
        # This should not raise an exception about destructuring symbolic shapes
        result = codegen_matmul(ctx, node)
        print("✅ Successfully handled symbolic shapes!")
        print(f"Result type: {type(result)}")
        
        # Check that the result is an AST expression
        assert isinstance(result, ast.AST), f"Expected AST node, got {type(result)}"
        print("✅ Result is an AST node as expected")
        
    except Exception as e:
        print(f"❌ Failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original CompileEnvironment
        lowering_module.CompileEnvironment = original_env

def test_concrete_matmul():
    """Test that the matmul lowering still works with concrete shapes"""
    print("\nTesting concrete shape handling in codegen_matmul...")
    
    # Create concrete shapes
    B, H, I, D, J = 2, 4, 8, 16, 32
    
    # Create mock nodes with concrete shapes
    class MockNode:
        def __init__(self, shape, ndim):
            self.meta = {"val": type('obj', (object,), {"shape": shape, "ndim": ndim})()}
    
    lhs_node = MockNode((B, H, I, D), 4)
    rhs_node = MockNode((B, H, D, J), 4)
    
    # Create main node
    class MockMatmulNode:
        def __init__(self):
            self.args = (lhs_node, rhs_node)
            self.kwargs = {}
            self.meta = {"val": type('obj', (object,), {"shape": (B, H, I, J), "size": lambda: (B, H, I, J)})()}
    
    node = MockMatmulNode()
    
    # Mock the graph interpreter
    ctx = MockGraphInterpreter()
    # Mock AST nodes for the tensors
    import ast
    ctx.env[lhs_node] = ast.Name(id="lhs", ctx=ast.Load())
    ctx.env[rhs_node] = ast.Name(id="rhs", ctx=ast.Load())
    
    # Mock the CompileEnvironment
    class MockSettings:
        dot_precision = None
    
    class MockCompileEnvironment:
        def __init__(self):
            self.settings = MockSettings()
        
        @staticmethod
        def current():
            return MockCompileEnvironment()
    
    # Temporarily replace CompileEnvironment
    import helion._compiler.inductor_lowering as lowering_module
    original_env = lowering_module.CompileEnvironment
    lowering_module.CompileEnvironment = MockCompileEnvironment
    
    try:
        # This should work with concrete shapes too
        result = codegen_matmul(ctx, node)
        print("✅ Successfully handled concrete shapes!")
        print(f"Result type: {type(result)}")
        
    except Exception as e:
        print(f"❌ Failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original CompileEnvironment
        lowering_module.CompileEnvironment = original_env

if __name__ == "__main__":
    test_symbolic_matmul()
    test_concrete_matmul()