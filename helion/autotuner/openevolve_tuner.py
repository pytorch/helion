"""
OpenEvolve-based Autotuner for Helion GPU Kernels
==================================================

This module implements an autotuner that uses OpenEvolve's evolutionary algorithm
to find optimal Helion kernel configurations. It serves as a drop-in replacement
for the differential evolution autotuner.

Example usage:
    from helion.autotuner.openevolve_tuner import OpenEvolveTuner

    config_space = {
        'block_size': [32, 64, 128, 256],
        'num_warps': [1, 2, 4, 8]
    }

    def objective(config):
        # Benchmark kernel with config
        return throughput_gbs  # Higher is better

    tuner = OpenEvolveTuner(config_space, objective, max_evaluations=100)
    best_config = tuner.tune()
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List

log = logging.getLogger(__name__)


class OpenEvolveTuner:
    """
    OpenEvolve-based autotuner for Helion GPU kernels.

    This class uses OpenEvolve's evolutionary algorithm to search for optimal
    kernel configurations. It converts Helion's config space into a format
    that OpenEvolve can evolve, runs the optimization, and returns the best
    configuration found.

    Attributes:
        config_space: Dictionary mapping parameter names to lists of valid values
        objective: Function that takes a config dict and returns a float (higher is better)
        max_evaluations: Maximum number of configurations to evaluate
        best_config: Best configuration found so far (None until tune() is called)
        best_score: Best score achieved so far (None until tune() is called)
    """

    def __init__(
        self,
        config_space: Dict[str, List[Any]],
        objective: Callable[[Dict[str, Any]], float],
        max_evaluations: int = 100,
        population_size: int = 20,
        temperature: float = 0.8,
        verbose: bool = True,
    ) -> None:
        """
        Initialize the OpenEvolveTuner.

        Args:
            config_space: Dict of tunable parameters with their valid values.
                Example: {
                    'block_size': [32, 64, 128, 256],
                    'num_warps': [1, 2, 4, 8],
                    'num_stages': [1, 2, 3, 4, 5]
                }
            objective: Function that takes config dict and returns float (higher is better).
                Should return 0.0 or -inf for invalid/failed configs.
            max_evaluations: Budget for tuning (number of iterations).
            population_size: Size of the population per island in OpenEvolve.
            temperature: LLM temperature for mutations (0.0-1.0).
            verbose: Whether to print progress information.
        """
        self._validate_config_space(config_space)

        self.config_space = config_space
        self.objective = objective
        self.max_evaluations = max_evaluations
        self.population_size = population_size
        self.temperature = temperature
        self.verbose = verbose

        self.best_config: Dict[str, Any] | None = None
        self.best_score: float | None = None
        self.evaluation_count = 0
        self.history: List[tuple[Dict[str, Any], float]] = []

    def _validate_config_space(self, config_space: Dict[str, List[Any]]) -> None:
        """Validate that the config space is well-formed."""
        if not config_space:
            raise ValueError("config_space cannot be empty")

        for param_name, values in config_space.items():
            if not isinstance(values, list):
                raise ValueError(
                    f"config_space['{param_name}'] must be a list, got {type(values)}"
                )
            if not values:
                raise ValueError(f"config_space['{param_name}'] cannot be empty")

    def _generate_initial_program(self) -> str:
        """
        Generate the initial program that OpenEvolve will evolve.

        This creates a Python function that returns a kernel configuration.
        OpenEvolve will mutate the values in this function to try different configs.
        """
        # Pick initial values (first value from each list)
        initial_values = {
            param: values[0] for param, values in self.config_space.items()
        }

        # Generate Python code
        lines = [
            "def get_kernel_config():",
            "    \"\"\"",
            "    Returns a kernel configuration dict.",
            "    OpenEvolve will evolve the values in this function.",
            "    \"\"\"",
            "    config = {",
        ]

        for param, value in initial_values.items():
            # Add comment showing valid values
            valid_values_str = str(self.config_space[param])
            lines.append(f"        # Valid values for {param}: {valid_values_str}")
            lines.append(f"        '{param}': {repr(value)},")

        lines.append("    }")
        lines.append("    return config")

        return "\n".join(lines)

    def _create_evaluator_function(self, evaluator_path: str) -> None:
        """
        Create the evaluator function that OpenEvolve will use.

        The evaluator:
        1. Imports the evolved program
        2. Calls get_kernel_config() to get the config
        3. Validates the config against config_space
        4. Calls the objective function
        5. Returns the score

        Args:
            evaluator_path: Path where the evaluator.py file will be written
        """
        # Create evaluator code
        evaluator_code = f"""
import sys
import importlib.util
import traceback
from pathlib import Path

# Config space for validation
CONFIG_SPACE = {repr(self.config_space)}

def validate_config(config):
    \"\"\"Check if config contains valid values from config_space.\"\"\"
    if not isinstance(config, dict):
        return False, "Config must be a dict"

    for param, value in config.items():
        if param not in CONFIG_SPACE:
            return False, f"Unknown parameter: {{param}}"

        if value not in CONFIG_SPACE[param]:
            return False, f"Invalid value {{value}} for {{param}}. Valid: {{CONFIG_SPACE[param]}}"

    # Check all required params are present
    for param in CONFIG_SPACE:
        if param not in config:
            return False, f"Missing required parameter: {{param}}"

    return True, None

def load_module_from_path(path):
    \"\"\"Load a Python module from a file path.\"\"\"
    spec = importlib.util.spec_from_file_location("evolved_program", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {{path}}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["evolved_program"] = module
    spec.loader.exec_module(module)
    return module

def evaluate(program_path):
    \"\"\"
    Evaluate an evolved program by extracting its config and running the objective.

    Returns:
        dict with 'score' key (higher is better)
    \"\"\"
    evaluation_count = {self.evaluation_count}

    try:
        # Load the evolved program
        program_module = load_module_from_path(program_path)

        # Extract the config
        if not hasattr(program_module, 'get_kernel_config'):
            if {self.verbose}:
                print(f"Evaluation {{evaluation_count}}: No get_kernel_config function found", file=sys.stderr)
            return {{"score": 0.0}}

        config = program_module.get_kernel_config()

        # Validate config
        is_valid, error_msg = validate_config(config)
        if not is_valid:
            if {self.verbose}:
                print(f"Evaluation {{evaluation_count}}: Invalid config: {{error_msg}}", file=sys.stderr)
            return {{"score": 0.0}}

        # Call the objective function (imported from the saved module)
        # We'll save the objective as a pickle file and load it
        import pickle
        with open('{evaluator_path}.objective.pkl', 'rb') as f:
            objective = pickle.load(f)

        # Evaluate
        score = objective(config)

        if {self.verbose}:
            print(f"Evaluation {{evaluation_count}}: config={{config}}, score={{score:.4f}}", file=sys.stderr)

        # Save history
        with open('{evaluator_path}.history.jsonl', 'a') as f:
            import json
            f.write(json.dumps({{'config': config, 'score': float(score)}}) + '\\n')

        return {{"score": float(score)}}

    except Exception as e:
        if {self.verbose}:
            print(f"Evaluation {{evaluation_count}}: Error: {{e}}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        return {{"score": 0.0}}
"""

        # Write evaluator
        Path(evaluator_path).write_text(evaluator_code)

        # Save objective function as pickle
        import pickle
        with open(f'{evaluator_path}.objective.pkl', 'wb') as f:
            pickle.dump(self.objective, f)

    def _create_config_yaml(self, config_path: str) -> None:
        """
        Create OpenEvolve configuration file.

        Args:
            config_path: Path where config.yaml will be written
        """
        # System message to guide the LLM
        system_message = """You are optimizing GPU kernel configurations for Helion.

OBJECTIVE: Find the configuration that maximizes throughput (GB/s or TFLOPS).

TUNABLE PARAMETERS:
""" + "\n".join([
            f"- {param}: Controls a kernel parameter. Valid values: {values}"
            for param, values in self.config_space.items()
        ]) + """

OPTIMIZATION STRATEGY:
1. Start with power-of-2 values (32, 64, 128, 256...) when applicable
2. Larger block sizes often help memory-bound kernels
3. More warps increase parallelism but have diminishing returns
4. Balance occupancy vs register pressure

CONSTRAINTS:
- All parameters must be from the allowed config_space
- Invalid configs will return 0.0 performance
- You can ONLY modify the values in get_kernel_config()
- Keep the function structure and return format unchanged

IMPORTANT: Only return values that are in the valid values list for each parameter.
"""

        config_yaml = f"""# OpenEvolve configuration for Helion kernel tuning
random_seed: 42
max_iterations: {self.max_evaluations}

llm:
  models:
    - name: "gpt-4o-mini"
      weight: 1.0
  temperature: {self.temperature}
  system_message: |
{chr(10).join('    ' + line for line in system_message.split(chr(10)))}

database:
  population_size: {self.population_size}
  num_islands: 3
  feature_dimensions: ["complexity", "diversity"]

evaluator:
  cascade_evaluation: true
  timeout: 60
"""

        Path(config_path).write_text(config_yaml)

    def tune(self) -> Dict[str, Any]:
        """
        Run OpenEvolve optimization to find the best config.

        Returns:
            Best configuration dictionary found during tuning.

        Raises:
            ImportError: If OpenEvolve is not installed
            RuntimeError: If tuning fails or no valid configs are found
        """
        try:
            from openevolve import run_evolution
        except ImportError as e:
            raise ImportError(
                "OpenEvolve is not installed. Install it with: pip install openevolve"
            ) from e

        if self.verbose:
            print(f"Starting OpenEvolve-based tuning with max_evaluations={self.max_evaluations}")
            print(f"Config space: {self.config_space}")

        # Create temporary directory for OpenEvolve files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Generate files
            initial_program_path = tmpdir_path / "initial_program.py"
            evaluator_path = tmpdir_path / "evaluator.py"
            config_path = tmpdir_path / "config.yaml"
            history_path = tmpdir_path / "evaluator.py.history.jsonl"

            initial_program = self._generate_initial_program()
            initial_program_path.write_text(initial_program)

            self._create_evaluator_function(str(evaluator_path))
            self._create_config_yaml(str(config_path))

            if self.verbose:
                print(f"\nInitial program:\n{initial_program}\n")

            # Set up environment variables for OpenAI API
            # Users need to set OPENAI_API_KEY
            if "OPENAI_API_KEY" not in os.environ:
                log.warning(
                    "OPENAI_API_KEY environment variable not set. "
                    "OpenEvolve requires an OpenAI API key to function. "
                    "Set it with: export OPENAI_API_KEY='your-key-here'"
                )

            try:
                # Run OpenEvolve
                if self.verbose:
                    print("\nStarting OpenEvolve optimization...")
                    print(f"This will make ~{self.max_evaluations} API calls to OpenAI.")
                    print(f"Estimated cost: $0.01-0.10 (depending on model and complexity)\n")

                result = run_evolution(
                    initial_program=str(initial_program_path),
                    evaluator=str(evaluator_path),
                    config=str(config_path),
                    iterations=self.max_evaluations,
                )

                if self.verbose:
                    print("\nOpenEvolve optimization complete!")

            except Exception as e:
                raise RuntimeError(
                    f"OpenEvolve optimization failed: {e}\n"
                    f"Check that OPENAI_API_KEY is set and valid."
                ) from e

            # Parse results from history
            if history_path.exists():
                history_data = []
                with open(history_path) as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        history_data.append((entry['config'], entry['score']))

                self.history = history_data

                # Find best config
                if history_data:
                    best_entry = max(history_data, key=lambda x: x[1])
                    self.best_config = best_entry[0]
                    self.best_score = best_entry[1]
                    self.evaluation_count = len(history_data)

            # Fallback: try to extract from result object
            if self.best_config is None and hasattr(result, 'best_code'):
                # Execute the best code to extract config
                try:
                    exec_globals: Dict[str, Any] = {}
                    exec(result.best_code, exec_globals)
                    if 'get_kernel_config' in exec_globals:
                        self.best_config = exec_globals['get_kernel_config']()
                        # Re-evaluate to get score
                        self.best_score = self.objective(self.best_config)
                except Exception as e:
                    log.warning(f"Failed to extract best config from result: {e}")

        # Final validation
        if self.best_config is None:
            # Fallback to random search baseline
            if self.verbose:
                print("\nWarning: OpenEvolve didn't find a valid config. Falling back to random search...")

            import random
            best_score = float('-inf')
            best_config = None

            for i in range(min(20, self.max_evaluations)):
                config = {
                    param: random.choice(values)
                    for param, values in self.config_space.items()
                }
                try:
                    score = self.objective(config)
                    if score > best_score:
                        best_score = score
                        best_config = config
                    if self.verbose:
                        print(f"Random evaluation {i+1}: config={config}, score={score:.4f}")
                except Exception as e:
                    if self.verbose:
                        print(f"Random evaluation {i+1}: Failed with error: {e}")

            if best_config is None:
                raise RuntimeError(
                    "No valid configuration found. All configs failed during evaluation."
                )

            self.best_config = best_config
            self.best_score = best_score

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"TUNING COMPLETE")
            print(f"{'='*60}")
            print(f"Best configuration: {self.best_config}")
            print(f"Best score: {self.best_score:.4f}")
            print(f"Total evaluations: {self.evaluation_count}")
            print(f"{'='*60}\n")

        return self.best_config
