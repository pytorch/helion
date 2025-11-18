# Repository Guidelines

This document explains how to work effectively in this repository.

## Project Structure & Module Organization

- `helion/`: Core Python package (DSL, compiler, runtime, autotuner).
- `test/`: PyTest test suite (`test_*.py`) with golden files `*.expected`.
- `examples/`: Runnable examples; each script must define a `main()` (pre-commit check).
- `docs/`: Sphinx docs; outputs to `site/`.
- `benchmarks/`, `dist/`, `scripts/`: Ancillary utilities and build artifacts.

## Build, Test, and Development Commands

- Lint/format/type-check:
  - `./lint.sh` (auto-formats and fixes some issues)
  - Optional hooks: `pre-commit run -a`
- Run tests: `pytest <filename>`
  - Run subset: `pytest <filename> -k name_substring`
- Build docs: `make -C docs html` (writes HTML under `site/`)

## Coding Style & Naming Conventions

- Language: Python 3.10+ with type hints; enable `from __future__ import annotations`.
- Formatting: Ruff formatter, line length 88, double quotes.
- Imports: Sorted by Ruff/isort; single import per line.
- Helion import pattern: `import helion; import helion.language as hl` (do not `import helion as hl`).
- Modules/files: snake_case; tests `test_*.py`; examples `*.py` with `main()`.
- Run `./lint.sh fix` before pushing; CI uses Ruff and Pyrefly.

## Testing Guidelines

- Framework: PyTest. Place tests in `test/` and name `test_<feature>.py`.
- Goldens: If using filecheck-style assertions, add a matching `test_<feature>.expected`.
- Use helpers in `helion._testing` (e.g., `check_example`, `TestCase.assertExpectedJournal`).
- Update goldens with `EXPECTTEST_ACCEPT=1`; validate generated code before correctness.
- Runtime: Many tests require CUDA, PyTorch nightly, and Triton dev builds; keep each test fast (<~30s).
- Local tips: For iteration, use `-k`, or set `HELION_USE_DEFAULT_CONFIG=1` to skip autotuning.
  - Warning: Do not run the full test suite with `HELION_USE_DEFAULT_CONFIG=1` — it can change execution paths and break tests. Only use this env var for targeted local iteration on specific tests.
- Helpful env vars: `HELION_LOGS=all|+all`, `HELION_PRINT_OUTPUT_CODE=1`, `HELION_USE_DEFAULT_CONFIG=1`.

### PyTest Debugging Tips

- Prefer `pytest -x -vv -s` while iterating: stops on first failure, prints full logs and stderr (needed to see generated code from failures that write to stderr).
- Target a single test via node id: `pytest test/test_examples.py::TestExamples::test_attention_block_pointer -x -vv -s`.
- For long outputs, pipe to a file: `... -s 2>&1 | tee /tmp/pytest.out`.
- Enable dtype/codegen checks when chasing codegen bugs: set `HELION_DEBUG_DTYPE_ASSERTS=1` and/or `HELION_PRINT_OUTPUT_CODE=1`.
- Show skip reasons with `pytest -ra`; narrow with `-k <pattern>` for fast cycles.

## Agent-Specific Instructions

- Do not run `pip install`, networked installs, or system package managers.
- Do not run `git commit`; users handle commits/branches.
- Do not `print()` inside kernels; use logging or host-side code.
- Tile indexing preserves dimensions; `i = hl.tile(...); x[i]` keeps ranks.
- Do not add unnecessary error checks via `hasattr`, `getattr`, `except`, etc.
