#!/usr/bin/env python3
"""
Script to automatically add # pyright: ignore comments based on actual pyright errors.
Runs pyright on a project and adds ignore comments to lines with reported errors.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import contextlib
import json
import operator
from pathlib import Path
import re
import subprocess
import sys


class PyrightError:
    def __init__(
        self,
        file_path: str,
        line: int,
        column: int,
        message: str,
        severity: str,
        rule: str | None = None,
    ):
        self.file_path = file_path
        self.line = line
        self.column = column
        self.message = message
        self.severity = severity
        self.rule = rule

    def __repr__(self):
        return f"PyrightError({self.file_path}:{self.line}:{self.column} - {self.rule or 'general'} - {self.message})"


class PyrightIgnoreAdder:
    def __init__(self, project_path: str = ".", pyright_config: str | None = None):
        """
        Initialize the adder for a specific project.

        Args:
            project_path: Path to the project root
            pyright_config: Path to pyright config file (optional)
        """
        self.project_path = Path(project_path)
        self.pyright_config = pyright_config
        self.errors: list[PyrightError] = []

    def run_pyright(self, output_format: str = "json") -> bool:
        """
        Run pyright on the project and collect errors.

        Args:
            output_format: Output format ('json' or 'text')

        Returns:
            True if pyright ran successfully, False otherwise
        """
        cmd = ["pyright"]

        if self.pyright_config:
            cmd.extend(["--project", self.pyright_config])

        if output_format == "json":
            cmd.append("--outputjson")

        cmd.append(str(self.project_path))

        try:
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_path
            )

            if output_format == "json":
                self._parse_json_output(result.stdout)
            else:
                self._parse_text_output(result.stdout)

            print(f"Found {len(self.errors)} pyright errors")
            return True

        except subprocess.CalledProcessError as e:
            print(f"Error running pyright: {e}")
            return False
        except FileNotFoundError:
            print(
                "Error: pyright not found. Please install pyright: npm install -g pyright"
            )
            return False

    def _parse_json_output(self, output: str) -> None:
        """Parse JSON output from pyright."""
        try:
            data = json.loads(output)

            for diagnostic in data.get("generalDiagnostics", []):
                file_path = diagnostic.get("file", "")
                range_info = diagnostic.get("range", {})
                start = range_info.get("start", {})
                line = start.get("line", 0) + 1  # Convert to 1-indexed
                column = start.get("character", 0) + 1  # Convert to 1-indexed
                message = diagnostic.get("message", "")
                severity = diagnostic.get("severity", "error")
                rule = diagnostic.get("rule")

                # Convert absolute path to relative path
                with contextlib.suppress(ValueError):
                    file_path = str(Path(file_path).relative_to(self.project_path))
                    # If we can't make it relative, use as-is

                error = PyrightError(file_path, line, column, message, severity, rule)
                self.errors.append(error)

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON output: {e}")
            print("Raw output:", output)

    def _parse_text_output(self, output: str) -> None:
        """Parse text output from pyright."""
        # Pattern to match pyright error lines
        # Example: src/main.py:10:5 - error: Cannot assign to variable (reportGeneralTypeIssues)
        pattern = r"^(.+?):(\d+):(\d+) - (error|warning|info): (.+?)(?:\s+\((.+?)\))?$"

        for line in output.splitlines():
            match = re.match(pattern, line.strip())
            if match:
                file_path, line_num, column, severity, message, rule = match.groups()

                # Convert absolute path to relative path
                with contextlib.suppress(ValueError):
                    file_path = str(Path(file_path).relative_to(self.project_path))
                    # If we can't make it relative, use as-is

                error = PyrightError(
                    file_path=file_path,
                    line=int(line_num),
                    column=int(column),
                    message=message,
                    severity=severity,
                    rule=rule,
                )
                self.errors.append(error)

    def group_errors_by_file(self) -> dict[str, list[PyrightError]]:
        """Group errors by file path."""
        errors_by_file = defaultdict(list)
        for error in self.errors:
            errors_by_file[error.file_path].append(error)
        return dict(errors_by_file)

    def add_ignore_comments(
        self,
        include_rule_names: bool = True,
        severity_filter: set[str] | None = None,
        dry_run: bool = False,
    ) -> dict[str, int]:
        """
        Add ignore comments to files based on pyright errors.

        Args:
            include_rule_names: Whether to include specific rule names in ignore comments
            severity_filter: Set of severities to include (e.g., {'error', 'warning'})
            dry_run: Preview changes without modifying files

        Returns:
            Dictionary mapping file paths to number of lines modified
        """
        if not self.errors:
            print("No errors found. Run pyright first.")
            return {}

        if severity_filter is None:
            severity_filter = {"error", "warning"}

        errors_by_file = self.group_errors_by_file()
        modified_files = {}

        for file_path, file_errors in errors_by_file.items():
            # Filter errors by severity
            filtered_errors = [e for e in file_errors if e.severity in severity_filter]
            if not filtered_errors:
                continue

            full_path = self.project_path / file_path

            if not full_path.exists():
                print(f"Warning: File not found: {full_path}")
                continue

            try:
                with open(full_path, encoding="utf-8") as f:
                    lines = f.readlines()

                # Group errors by line number
                errors_by_line = defaultdict(list)
                for error in filtered_errors:
                    errors_by_line[error.line].append(error)

                # Add ignore comments (process from bottom to top to avoid line number shifts)
                modified_lines = 0
                for line_num in sorted(errors_by_line.keys(), reverse=True):
                    line_errors = errors_by_line[line_num]

                    if line_num > len(lines):
                        print(f"Warning: Line {line_num} not found in {file_path}")
                        continue

                    line_idx = line_num - 1  # Convert to 0-indexed
                    current_line = lines[line_idx].rstrip("\n")

                    # Skip if already has pyright ignore
                    if "pyright: ignore" in current_line:
                        continue

                    # Create ignore comment
                    ignore_comment = self._create_ignore_comment(
                        line_errors, include_rule_names
                    )

                    # Add ignore comment
                    if current_line.strip():  # Only add to non-empty lines
                        new_line = f"{current_line}  {ignore_comment}\n"
                        lines[line_idx] = new_line
                        modified_lines += 1

                if modified_lines > 0:
                    if dry_run:
                        print(f"Would modify {modified_lines} lines in {file_path}")
                        for line_num in sorted(errors_by_line.keys()):
                            line_errors = errors_by_line[line_num]
                            print(f"  Line {line_num}: {len(line_errors)} error(s)")
                            for error in line_errors:
                                print(
                                    f"    - {error.rule or 'general'}: {error.message}"
                                )
                    else:
                        with open(full_path, "w", encoding="utf-8") as f:
                            f.writelines(lines)
                        print(f"Modified {modified_lines} lines in {file_path}")

                    modified_files[file_path] = modified_lines

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        return modified_files

    def _create_ignore_comment(
        self, errors: list[PyrightError], include_rule_names: bool
    ) -> str:
        """Create an appropriate ignore comment for a list of errors."""
        if not include_rule_names:
            return "# pyright: ignore"

        # Collect unique rule names
        rules = set()
        for error in errors:
            if error.rule:
                rules.add(error.rule)

        if rules:
            rules_str = ",".join(sorted(rules))
            return f"# pyright: ignore[{rules_str}]"
        return "# pyright: ignore"

    def show_error_summary(self) -> None:
        """Display a summary of found errors."""
        if not self.errors:
            print("No errors found.")
            return

        # Group by severity
        by_severity = defaultdict(int)
        by_file = defaultdict(int)
        by_rule = defaultdict(int)

        for error in self.errors:
            by_severity[error.severity] += 1
            by_file[error.file_path] += 1
            if error.rule:
                by_rule[error.rule] += 1

        print("\n=== Error Summary ===")
        print(f"Total errors: {len(self.errors)}")

        print("\nBy severity:")
        for severity, count in sorted(by_severity.items()):
            print(f"  {severity}: {count}")

        print("\nBy file (top 10):")
        for file_path, count in sorted(
            by_file.items(), key=operator.itemgetter(1), reverse=True
        )[:10]:
            print(f"  {file_path}: {count}")

        if by_rule:
            print("\nBy rule (top 10):")
            for rule, count in sorted(
                by_rule.items(), key=operator.itemgetter(1), reverse=True
            )[:10]:
                print(f"  {rule}: {count}")

    def export_errors(self, output_file: str) -> None:
        """Export errors to a JSON file."""
        errors_data = []
        for error in self.errors:
            errors_data.append(
                {
                    "file": error.file_path,
                    "line": error.line,
                    "column": error.column,
                    "message": error.message,
                    "severity": error.severity,
                    "rule": error.rule,
                }
            )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(errors_data, f, indent=2)

        print(f"Exported {len(errors_data)} errors to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run pyright and automatically add ignore comments to error lines"
    )
    parser.add_argument(
        "--project",
        default=".",
        help="Path to project root (default: current directory)",
    )
    parser.add_argument("--config", help="Path to pyright config file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )
    parser.add_argument(
        "--no-rule-names",
        action="store_true",
        help="Don't include specific rule names in ignore comments",
    )
    parser.add_argument(
        "--severity",
        choices=["error", "warning", "info"],
        nargs="*",
        default=["error", "warning"],
        help="Severities to include (default: error, warning)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only show error summary, don't add ignore comments",
    )
    parser.add_argument("--export-errors", help="Export errors to JSON file")
    parser.add_argument(
        "--output-format",
        choices=["json", "text"],
        default="json",
        help="Pyright output format (default: json)",
    )

    args = parser.parse_args()

    # Initialize the adder
    adder = PyrightIgnoreAdder(args.project, args.config)

    # Run pyright
    if not adder.run_pyright(args.output_format):
        sys.exit(1)

    # Show summary
    adder.show_error_summary()

    # Export errors if requested
    if args.export_errors:
        adder.export_errors(args.export_errors)

    # Add ignore comments if not summary-only
    if not args.summary_only:
        severity_filter = set(args.severity)
        include_rule_names = not args.no_rule_names

        modified_files = adder.add_ignore_comments(
            include_rule_names=include_rule_names,
            severity_filter=severity_filter,
            dry_run=args.dry_run,
        )

        if not args.dry_run and modified_files:
            print(f"\nSuccessfully modified {len(modified_files)} files")
            print("You may want to run pyright again to verify the changes")


if __name__ == "__main__":
    main()


# Example usage:
"""
# Run pyright and add ignore comments to all error lines
python scripts/insert-pyright-ignore.py

# Dry run to see what would be changed
python scripts/insert-pyright-ignore.py --dry-run

# Only show error summary
python scripts/insert-pyright-ignore.py --summary-only

# Only add ignore comments for errors (not warnings)
python scripts/insert-pyright-ignore.py --severity error

# Don't include specific rule names in ignore comments
python scripts/insert-pyright-ignore.py --no-rule-names

# Use specific project directory and config
python scripts/insert-pyright-ignore.py --project /path/to/project --config pyrightconfig.json

# Export errors to JSON file
python scripts/insert-pyright-ignore.py --export-errors errors.json

# Use text output format instead of JSON
python scripts/insert-pyright-ignore.py --output-format text
"""
