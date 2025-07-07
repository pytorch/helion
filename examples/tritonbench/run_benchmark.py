"""Generic runner for Helion kernels in tritonbench.

Usage:
    python run_helion_benchmark.py --kernel vector_add [tritonbench args...]
"""

import sys
import os
import argparse
import importlib
import subprocess


def check_and_setup_tritonbench():
    """Check if tritonbench is properly initialized and installed."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    helion_root = os.path.abspath(os.path.join(script_dir, "../.."))
    tritonbench_path = os.path.join(helion_root, "third_party/tritonbench")
    
    # Check if tritonbench directory exists and has content
    if not os.path.exists(tritonbench_path) or not os.listdir(tritonbench_path):
        print("Tritonbench submodule not initialized. Initializing and installing...")
        try:
            # Run both commands together
            # First, initialize submodule
            subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive"],
                cwd=helion_root,
                check=True
            )
            
            # Then run install.py
            original_dir = os.getcwd()
            os.chdir(tritonbench_path)
            subprocess.run([sys.executable, "install.py"], check=True)
            os.chdir(original_dir)
            
            print("Tritonbench setup completed successfully.")
                
        except subprocess.CalledProcessError as e:
            print(f"Error setting up tritonbench: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error during setup: {e}")
            sys.exit(1)
    
    # Add to path
    if tritonbench_path not in sys.path:
        sys.path.insert(0, tritonbench_path)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Helion kernels with tritonbench")
    parser.add_argument(
        "--kernel",
        type=str,
        required=True,
        help="Name of the Helion kernel module (e.g., vector_add)"
    )
    
    # Parse known args to get the kernel name, pass rest to tritonbench
    args, tritonbench_args = parser.parse_known_args()
    
    # Check and setup tritonbench if needed
    check_and_setup_tritonbench()
    
    # Add necessary paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    
    # Add helion root to path
    helion_root = os.path.abspath(os.path.join(script_dir, "../.."))
    if helion_root not in sys.path:
        sys.path.insert(0, helion_root)
    
    # Add tritonbench to path if needed (do this early)
    tritonbench_path = os.path.join(helion_root, "third_party/tritonbench")
    if os.path.exists(tritonbench_path):
        sys.path.insert(0, tritonbench_path)
    
    # Import the kernel module
    try:
        kernel_module = importlib.import_module(args.kernel)
    except ImportError as e:
        print(f"Error: Could not import kernel module '{args.kernel}'")
        print(f"Make sure {args.kernel}.py exists in {script_dir}")
        print(f"Import error: {e}")
        sys.exit(1)
    
    # Infer kernel information from module
    # The kernel name is the module name
    kernel_name = args.kernel
    
    # Find the kernel function in the module (function with same name as module)
    if not hasattr(kernel_module, kernel_name):
        print(f"Error: Module '{args.kernel}' does not have a kernel function named '{kernel_name}'")
        sys.exit(1)
    
    kernel_func = getattr(kernel_module, kernel_name)
    
    # Import tritonbench components
    try:
        from tritonbench.utils.parser import get_parser
        from tritonbench.operators_collection import list_operators_by_collection
    except ImportError:
        print("Error: Could not import tritonbench. Make sure it's in the path.")
        sys.exit(1)
    
    # Get the tritonbench operator name (assume it's the same as the kernel name)
    operator_name = kernel_name
    
    # Parse tritonbench arguments
    tb_parser = get_parser()
    
    assert '--op' not in tritonbench_args
    tritonbench_args = ['--op', operator_name] + tritonbench_args
    
    tb_args = tb_parser.parse_args(tritonbench_args)
    
    # Register the Helion kernel with tritonbench BEFORE importing the operator
    from tritonbench.utils.triton_op import register_benchmark_mannually, REGISTERED_BENCHMARKS
    
    # Create the benchmark method
    def create_helion_method(kernel_func):
        def helion_method(self, *args):
            """Helion implementation."""
            def _inner():
                return kernel_func(*args)
            return _inner
        return helion_method
    
    # Register it as a benchmark first
    helion_method_name = f"helion_{kernel_name}"
    register_benchmark_mannually(
        operator_name=operator_name,
        func_name=helion_method_name,
        baseline=False,
        enabled=True,
        label=helion_method_name
    )
    
    # Import and run the operator
    operator_module_name = f"tritonbench.operators.{operator_name}.operator"
    try:
        operator_module = importlib.import_module(operator_module_name)
        Operator = operator_module.Operator
    except ImportError:
        print(f"Error: Could not import operator '{operator_name}' from tritonbench")
        sys.exit(1)
    
    # Monkey-patch the Operator class after import
    setattr(Operator, helion_method_name, create_helion_method(kernel_func))
    
    print(f"Running {operator_name} benchmark with Helion implementation...\n")
    
    # Create and run the operator
    op = Operator(tb_args=tb_args, extra_args={})
    
    # Run with proper parameters
    warmup = getattr(tb_args, 'warmup', 25)
    rep = getattr(tb_args, 'iter', 100)
    op.run(warmup=warmup, rep=rep)
    
    # Print results
    print("\nBenchmark Results:")
    print(op.output)

if __name__ == "__main__":
    main()