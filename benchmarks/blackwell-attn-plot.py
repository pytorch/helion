#!/usr/bin/env python3
"""
Terminal bar plot generator from CSV files.
No dependencies required - uses only Python standard library.

Usage:
    python plot_bars.py data.csv --x col1 --legend col2 --value col3
    python plot_bars.py data.csv --x col1,col2 --legend col3 --value col4 --colors key1=red,key2=blue
    python plot_bars.py data.csv --x col1 --legend col2 --value col3 --vertical
    python plot_bars.py data.csv --x col1 --legend col2 --value col3 --patterns key1=/,key2=\\
    python plot_bars.py data.csv --x col1 --legend col2 --value col3 --rename old_name=new_name
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import sys

# Unicode block characters for drawing bars
BLOCKS = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]

# Vertical bar characters
VERTICAL_BLOCKS = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]

# Pattern characters for bars
PATTERNS = {
    "/": "/",
    "\\": "\\",
    "x": "x",
    "+": "+",
    "-": "-",
    "|": "|",
    ".": ".",
    "o": "o",
    "*": "*",
    "#": "#",
    "=": "=",
}

# ANSI color codes
COLORS = {
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
    "white": "\033[97m",
    "gray": "\033[90m",
    "reset": "\033[0m",
}

# Default color palette
DEFAULT_PALETTE = ["blue", "green", "red", "yellow", "magenta", "cyan", "white", "gray"]


def parse_csv(filename: str) -> tuple[list[str], list[dict[str, str]]]:
    """Parse CSV file and return headers and rows."""
    with open(filename) as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)
    return headers, rows


def make_tuple(row: dict[str, str], columns: list[str]) -> tuple[str, ...]:
    """Create a tuple from specified columns."""
    return tuple(row[col] for col in columns)


def aggregate_data(
    rows: list[dict[str, str]],
    x_cols: list[str],
    legend_cols: list[str] | None,
    value_col: str,
) -> dict[tuple, dict[tuple, list[float]]]:
    """
    Aggregate data by x-axis and legend keys.
    Returns: {x_key: {legend_key: [values]}}
    """
    data = defaultdict(lambda: defaultdict(list))

    for row in rows:
        x_key = make_tuple(row, x_cols)

        if legend_cols:
            legend_key = make_tuple(row, legend_cols)
        else:
            legend_key = ("",)

        try:
            value = float(row[value_col])
        except (ValueError, KeyError):
            continue

        data[x_key][legend_key].append(value)

    return data


def format_tuple(t: tuple[str, ...]) -> str:
    """Format tuple for display."""
    if len(t) == 1:
        return t[0]
    return "(" + ", ".join(t) + ")"


def draw_bar(
    value: float, max_value: float, width: int, color: str, pattern: str | None = None
) -> str:
    """Draw a single bar using Unicode block characters."""
    if max_value == 0:
        return ""

    # Calculate bar length in characters
    ratio = value / max_value
    full_blocks = int(ratio * width)
    remainder = (ratio * width - full_blocks) * 8
    partial_block_idx = int(remainder)

    # Build the bar
    if pattern and pattern in PATTERNS:
        # Use pattern character instead of block
        bar = PATTERNS[pattern] * full_blocks
        if full_blocks < width and partial_block_idx > 0:
            bar += BLOCKS[partial_block_idx]
    else:
        bar = BLOCKS[-1] * full_blocks
        if full_blocks < width and partial_block_idx > 0:
            bar += BLOCKS[partial_block_idx]

    # Apply color
    color_code = COLORS.get(color, "")
    reset_code = COLORS["reset"] if color_code else ""

    return f"{color_code}{bar}{reset_code}"


def apply_exclusions(
    data: dict[tuple, dict[tuple, list[float]]], exclude_list: list[str]
) -> dict[tuple, dict[tuple, list[float]]]:
    """Filter out excluded series from data."""
    if not exclude_list:
        return data

    filtered_data = {}
    for x_key, legend_dict in data.items():
        filtered_dict = {}
        for legend_key, values in legend_dict.items():
            key_str = format_tuple(legend_key)
            if key_str not in exclude_list:
                filtered_dict[legend_key] = values
        if filtered_dict:  # Only keep x_key if it has data after filtering
            filtered_data[x_key] = filtered_dict

    return filtered_data


def apply_combinations(
    data: dict[tuple, dict[tuple, list[float]]],
    combinations: list[tuple[str, list[str], str]],
) -> tuple[dict[tuple, dict[tuple, list[float]]], dict[tuple, str]]:
    """
    Combine multiple series into one using specified aggregation.
    combinations: List of (new_name, [series_to_combine], aggregation_method)
    Returns: (modified_data, aggregation_map) where aggregation_map maps combined series to their agg method
    """
    if not combinations:
        return data, {}

    import statistics

    new_data = {}
    agg_map = {}  # Maps combined series keys to their aggregation method

    for x_key, legend_dict in data.items():
        new_dict = dict(legend_dict)  # Copy existing data

        for new_name, series_names, agg_method in combinations:
            # Collect values from each series separately for proper aggregation
            series_values = []  # List of lists, one per series
            keys_to_remove = []

            for legend_key, values in legend_dict.items():
                key_str = format_tuple(legend_key)
                if key_str in series_names:
                    series_values.append(values)
                    keys_to_remove.append(legend_key)

            if series_values:
                # Remove the original series
                for key in keys_to_remove:
                    new_dict.pop(key, None)

                # Combine based on aggregation method
                # For each x position, we need to aggregate across the series
                new_key = (new_name,)

                # Get max length (some series might have different numbers of values)
                max_len = max(len(vals) for vals in series_values)

                combined_values = []
                for i in range(max_len):
                    # Get all values at position i across all series
                    point_values = []
                    for vals in series_values:
                        if i < len(vals):
                            point_values.append(vals[i])

                    if point_values:
                        if agg_method == "min":
                            combined_values.append(min(point_values))
                        elif agg_method == "max":
                            combined_values.append(max(point_values))
                        elif agg_method == "mean":
                            combined_values.append(statistics.mean(point_values))
                        elif agg_method == "sum":
                            combined_values.append(sum(point_values))
                        else:
                            combined_values.append(statistics.mean(point_values))

                new_dict[new_key] = combined_values
                agg_map[new_key] = agg_method

        new_data[x_key] = new_dict

    return new_data, agg_map


def aggregate_values(
    data: dict[tuple, dict[tuple, list[float]]], agg_method: str = "sum"
) -> dict[tuple, dict[tuple, float]]:
    """
    Aggregate lists of values to single values using specified method.
    """
    import statistics

    aggregated = {}
    for x_key, legend_dict in data.items():
        agg_dict = {}
        for legend_key, values in legend_dict.items():
            if not values:
                agg_dict[legend_key] = 0.0
            elif agg_method == "sum":
                agg_dict[legend_key] = sum(values)
            elif agg_method == "mean":
                agg_dict[legend_key] = statistics.mean(values)
            elif agg_method == "min":
                agg_dict[legend_key] = min(values)
            elif agg_method == "max":
                agg_dict[legend_key] = max(values)
            else:
                agg_dict[legend_key] = sum(values)  # Default to sum
        aggregated[x_key] = agg_dict

    return aggregated


def apply_renames(
    legend_keys: list[tuple], renames: dict[str, str]
) -> dict[tuple, str]:
    """Create a mapping of original keys to renamed display strings."""
    rename_map = {}
    for key in legend_keys:
        key_str = format_tuple(key)
        rename_map[key] = renames.get(key_str, key_str)
    return rename_map


def assign_colors(
    legend_keys: list[tuple], custom_colors: dict[str, str]
) -> dict[tuple, str]:
    """Assign colors to legend keys."""
    color_map = {}
    palette_idx = 0

    for key in legend_keys:
        key_str = format_tuple(key)

        # Check if custom color is specified
        if key_str in custom_colors:
            color_map[key] = custom_colors[key_str]
        else:
            # Auto-assign from palette
            color_map[key] = DEFAULT_PALETTE[palette_idx % len(DEFAULT_PALETTE)]
            palette_idx += 1

    return color_map


def assign_patterns(
    legend_keys: list[tuple], custom_patterns: dict[str, str]
) -> dict[tuple, str | None]:
    """Assign patterns to legend keys."""
    pattern_map = {}

    for key in legend_keys:
        key_str = format_tuple(key)

        # Check if custom pattern is specified
        if key_str in custom_patterns:
            pattern_map[key] = custom_patterns[key_str]
        else:
            pattern_map[key] = None

    return pattern_map


def plot_bars(
    data: dict[tuple, dict[tuple, float]],
    x_cols: list[str],
    legend_cols: list[str] | None,
    value_col: str,
    custom_colors: dict[str, str],
    custom_patterns: dict[str, str],
    renames: dict[str, str],
    bar_width: int = 50,
    show_values: bool = True,
):
    """Generate and print terminal bar plot."""

    # Get all legend keys and assign colors, patterns, and renames
    all_legend_keys = set()
    for legend_dict in data.values():
        all_legend_keys.update(legend_dict.keys())
    legend_keys = sorted(all_legend_keys)
    color_map = assign_colors(legend_keys, custom_colors)
    pattern_map = assign_patterns(legend_keys, custom_patterns)
    rename_map = apply_renames(legend_keys, renames)

    # Find max value for scaling
    max_value = 0
    for legend_dict in data.values():
        for value in legend_dict.values():
            max_value = max(max_value, value)

    # Calculate label width
    x_label = "+".join(x_cols)
    max_x_label_len = max(len(format_tuple(k)) for k in data)
    label_width = max(len(x_label), max_x_label_len)

    # Print title
    print(
        f"\n{value_col} by {x_label}"
        + (f" (grouped by {'+'.join(legend_cols)})" if legend_cols else "")
    )
    print("=" * (label_width + bar_width + 20))
    print()

    # Print legend if there are multiple series
    if len(legend_keys) > 1 or (len(legend_keys) == 1 and legend_keys[0] != ("",)):
        print("Legend:")
        for key in legend_keys:
            display_name = rename_map[key]
            color = color_map[key]
            pattern = pattern_map[key]
            color_code = COLORS.get(color, "")
            reset_code = COLORS["reset"]
            # Show pattern in legend if applicable
            legend_char = PATTERNS.get(pattern, "█") if pattern else "█"
            print(f"  {color_code}{legend_char}{reset_code} {display_name}")
        print()

    # Sort x-axis keys
    sorted_x_keys = sorted(data.keys())

    # Print bars
    for x_key in sorted_x_keys:
        x_label = format_tuple(x_key)
        print(f"{x_label:<{label_width}} │", end="")

        legend_dict = data[x_key]

        # If multiple legend keys, stack them horizontally with separators
        if len(legend_keys) > 1:
            print()
            for legend_key in legend_keys:
                value = legend_dict.get(legend_key, 0)
                if (
                    value > 0 or len(legend_keys) <= 3
                ):  # Show empty bars if few categories
                    color = color_map[legend_key]
                    pattern = pattern_map[legend_key]
                    bar = draw_bar(value, max_value, bar_width, color, pattern)
                    legend_label = rename_map[legend_key]
                    value_str = f" {value:.2f}" if show_values else ""
                    print(f"{' ' * label_width} │ {bar}{value_str}")
        else:
            # Single series - show inline
            legend_key = legend_keys[0] if legend_keys else ("",)
            value = legend_dict.get(legend_key, 0)
            color = color_map.get(legend_key, "white")
            pattern = pattern_map.get(legend_key)
            bar = draw_bar(value, max_value, bar_width, color, pattern)
            value_str = f" {value:.2f}" if show_values else ""
            print(f" {bar}{value_str}")

    print()
    print(f"Max value: {max_value:.2f}")
    print()


def plot_bars_vertical(
    data: dict[tuple, dict[tuple, float]],
    x_cols: list[str],
    legend_cols: list[str] | None,
    value_col: str,
    custom_colors: dict[str, str],
    custom_patterns: dict[str, str],
    renames: dict[str, str],
    bar_height: int = 20,
    show_values: bool = True,
):
    """Generate and print vertical terminal bar plot."""

    # Get all legend keys and assign colors, patterns, and renames
    all_legend_keys = set()
    for legend_dict in data.values():
        all_legend_keys.update(legend_dict.keys())
    legend_keys = sorted(all_legend_keys)
    color_map = assign_colors(legend_keys, custom_colors)
    pattern_map = assign_patterns(legend_keys, custom_patterns)
    rename_map = apply_renames(legend_keys, renames)

    # Find max value for scaling
    max_value = 0
    for legend_dict in data.values():
        for value in legend_dict.values():
            max_value = max(max_value, value)

    # Print title
    x_label = "+".join(x_cols)
    print(
        f"\n{value_col} by {x_label}"
        + (f" (grouped by {'+'.join(legend_cols)})" if legend_cols else "")
    )
    print("=" * 80)
    print()

    # Print legend if there are multiple series
    if len(legend_keys) > 1 or (len(legend_keys) == 1 and legend_keys[0] != ("",)):
        print("Legend:")
        for key in legend_keys:
            display_name = rename_map[key]
            color = color_map[key]
            pattern = pattern_map[key]
            color_code = COLORS.get(color, "")
            reset_code = COLORS["reset"]
            legend_char = PATTERNS.get(pattern, "█") if pattern else "█"
            print(f"  {color_code}{legend_char}{reset_code} {display_name}")
        print()

    # Sort x-axis keys
    sorted_x_keys = sorted(data.keys())

    # Calculate column width for labels
    max_label_width = max(len(format_tuple(k)) for k in sorted_x_keys)

    # For each legend key, we need a column (or group of columns if showing values)
    num_x_positions = len(sorted_x_keys)
    num_series = len(legend_keys)

    # Column width should accommodate both the label and the bars
    # When multiple series, each series gets 1 character, plus extra spacing between groups
    min_width_for_bars = num_series * 2 if num_series > 1 else 3
    col_width = max(
        max_label_width, min_width_for_bars, 10
    )  # At least 10 chars for spacing

    # Print the bars from top to bottom
    for row in range(bar_height, -1, -1):
        line_parts = []
        for x_key in sorted_x_keys:
            legend_dict = data[x_key]

            # Determine which legend keys to show
            if len(legend_keys) == 1:
                # Single series
                legend_key = legend_keys[0]
                value = legend_dict.get(legend_key, 0)
                ratio = value / max_value if max_value > 0 else 0
                height = ratio * bar_height

                color = color_map[legend_key]
                pattern = pattern_map[legend_key]
                color_code = COLORS.get(color, "")
                reset_code = COLORS["reset"]

                if row == 0:
                    # Bottom row - show x-axis label
                    label = format_tuple(x_key)
                    line_parts.append(f"{label:^{col_width}}")
                elif row <= height:
                    # Show bar character
                    # Calculate which block character to use
                    if abs(row - height) < 1 and height % 1 > 0:
                        # Partial block at the top
                        block_idx = int((height % 1) * 8)
                        char = (
                            VERTICAL_BLOCKS[block_idx]
                            if not pattern
                            else PATTERNS.get(pattern, "█")
                        )
                    else:
                        # Full block
                        char = (
                            VERTICAL_BLOCKS[-1]
                            if not pattern
                            else PATTERNS.get(pattern, "█")
                        )
                    # Center the character manually to avoid color code length issues
                    padding = (col_width - 1) // 2
                    line_parts.append(
                        " " * padding
                        + f"{color_code}{char}{reset_code}"
                        + " " * (col_width - 1 - padding)
                    )
                else:
                    # Empty space
                    line_parts.append(" " * col_width)
            else:
                # Multiple series - show them side by side
                chars = []
                for legend_key in legend_keys:
                    value = legend_dict.get(legend_key, 0)
                    ratio = value / max_value if max_value > 0 else 0
                    height = ratio * bar_height

                    color = color_map[legend_key]
                    pattern = pattern_map[legend_key]
                    color_code = COLORS.get(color, "")
                    reset_code = COLORS["reset"]

                    if row == 0:
                        continue  # Handle labels separately
                    if row <= height:
                        if abs(row - height) < 1 and height % 1 > 0:
                            block_idx = int((height % 1) * 8)
                            char = (
                                VERTICAL_BLOCKS[block_idx]
                                if not pattern
                                else PATTERNS.get(pattern, "█")
                            )
                        else:
                            char = (
                                VERTICAL_BLOCKS[-1]
                                if not pattern
                                else PATTERNS.get(pattern, "█")
                            )
                        chars.append(f"{color_code}{char}{reset_code}")
                    else:
                        chars.append(" ")

                if row == 0:
                    label = format_tuple(x_key)
                    line_parts.append(f"{label:^{col_width}}")
                else:
                    # Add spacing between bars within a group
                    bars_with_spacing = " ".join(chars)
                    # Calculate the visual width (number of actual characters, not counting color codes)
                    visual_width = len(legend_keys) + (
                        len(legend_keys) - 1
                    )  # chars + spaces between
                    padding = (col_width - visual_width) // 2
                    line_parts.append(
                        " " * padding
                        + bars_with_spacing
                        + " " * (col_width - visual_width - padding)
                    )

        print("  ".join(line_parts))

    # Print values if requested
    if show_values:
        print()
        for legend_key in legend_keys:
            display_name = rename_map[legend_key]
            values_line = [display_name[:col_width].ljust(col_width)]
            for x_key in sorted_x_keys:
                value = data[x_key].get(legend_key, 0)
                values_line.append(f"{value:.2f}".center(col_width))
            print("  ".join(values_line))

    print()
    print(f"Max value: {max_value:.2f}")
    print()


def parse_color_spec(color_spec: str) -> dict[str, str]:
    """Parse color specification like 'key1=red,key2=blue'."""
    colors = {}
    if not color_spec:
        return colors

    for pair in color_spec.split(","):
        if "=" not in pair:
            continue
        key, color = pair.split("=", 1)
        colors[key.strip()] = color.strip()

    return colors


def parse_pattern_spec(pattern_spec: str) -> dict[str, str]:
    """Parse pattern specification like 'key1=/,key2=\\'."""
    patterns = {}
    if not pattern_spec:
        return patterns

    for pair in pattern_spec.split(","):
        if "=" not in pair:
            continue
        key, pattern = pair.split("=", 1)
        patterns[key.strip()] = pattern.strip()

    return patterns


def parse_rename_spec(rename_spec: str) -> dict[str, str]:
    """Parse rename specification like 'old1=new1,old2=new2'."""
    renames = {}
    if not rename_spec:
        return renames

    for pair in rename_spec.split(","):
        if "=" not in pair:
            continue
        old_name, new_name = pair.split("=", 1)
        renames[old_name.strip()] = new_name.strip()

    return renames


def parse_exclude_spec(exclude_spec: str) -> list[str]:
    """Parse exclude specification like 'series1,series2,series3'."""
    if not exclude_spec:
        return []
    return [name.strip() for name in exclude_spec.split(",")]


def parse_combine_spec(combine_spec: str) -> list[tuple[str, list[str], str]]:
    """
    Parse combine specification like 'new_name=series1+series2:min,another=s3+s4:max'.
    Returns: List of (new_name, [series_to_combine], aggregation_method)
    """
    combinations = []
    if not combine_spec:
        return combinations

    for combo in combine_spec.split(","):
        if "=" not in combo:
            continue

        new_name, rest = combo.split("=", 1)
        new_name = new_name.strip()

        # Check if aggregation method is specified
        if ":" in rest:
            series_part, agg_method = rest.rsplit(":", 1)
            agg_method = agg_method.strip().lower()
            if agg_method not in ["min", "max", "mean", "sum"]:
                print(
                    f"Warning: Unknown aggregation method '{agg_method}', using 'mean'",
                    file=sys.stderr,
                )
                agg_method = "mean"
        else:
            series_part = rest
            agg_method = "mean"  # Default to mean

        # Parse series names (split by +)
        series_names = [s.strip() for s in series_part.split("+")]

        if len(series_names) < 2:
            print(
                f"Warning: Combination '{combo}' needs at least 2 series to combine",
                file=sys.stderr,
            )
            continue

        combinations.append((new_name, series_names, agg_method))

    return combinations


def main():
    parser = argparse.ArgumentParser(
        description="Generate terminal bar plots from CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data.csv --x model --value accuracy
  %(prog)s data.csv --x model --legend dataset --value score
  %(prog)s data.csv --x model,size --legend dataset --value time --colors gpt=blue,llama=red
  %(prog)s data.csv --x model --value throughput --width 60 --no-values
  %(prog)s data.csv --x model --legend dataset --value score --vertical
  %(prog)s data.csv --x model --legend dataset --value score --patterns dataset1=/,dataset2=\\
  %(prog)s data.csv --x model --legend dataset --value score --rename old_name=New Name
  %(prog)s data.csv --x model --legend dataset --value score --exclude unwanted_series
  %(prog)s data.csv --x model --legend dataset --value score --combine "Best=A+B:min,Worst=C+D:max"

Available colors: red, green, yellow, blue, magenta, cyan, white, gray
Available patterns: /, \\, x, +, -, |, ., o, *, #, =
Aggregation methods for --combine: min, max, mean, sum (default: mean)
        """,
    )

    parser.add_argument("csv_file", help="CSV file to read")
    parser.add_argument(
        "--x",
        required=True,
        help="Comma-separated column(s) for x-axis (creates tuple if multiple)",
    )
    parser.add_argument(
        "--legend",
        help="Comma-separated column(s) for legend/color grouping (creates tuple if multiple)",
    )
    parser.add_argument(
        "--value", required=True, help="Column containing the numeric values to plot"
    )
    parser.add_argument(
        "--colors", help="Custom color mapping: key1=color1,key2=color2,..."
    )
    parser.add_argument(
        "--patterns",
        help="Custom pattern mapping: key1=pattern1,key2=pattern2,... (e.g., key1=/,key2=\\)",
    )
    parser.add_argument(
        "--rename", help="Rename legend entries: old1=new1,old2=new2,..."
    )
    parser.add_argument(
        "--exclude", help="Exclude series from plot: series1,series2,..."
    )
    parser.add_argument(
        "--combine",
        help="Combine series: new_name=series1+series2:agg_method (agg_method: min, max, mean, sum). Example: Combined=A+B:min,Average=C+D:mean",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=50,
        help="Width of bars in characters for horizontal bars (default: 50)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=20,
        help="Height of bars in characters for vertical bars (default: 20)",
    )
    parser.add_argument(
        "--vertical",
        action="store_true",
        help="Create vertical bars instead of horizontal",
    )
    parser.add_argument(
        "--no-values", action="store_true", help="Hide numeric values next to bars"
    )

    args = parser.parse_args()

    # Parse column specifications
    x_cols = [col.strip() for col in args.x.split(",")]
    legend_cols = (
        [col.strip() for col in args.legend.split(",")] if args.legend else None
    )
    value_col = args.value
    custom_colors = parse_color_spec(args.colors)
    custom_patterns = parse_pattern_spec(args.patterns)
    renames = parse_rename_spec(args.rename)
    exclude_list = parse_exclude_spec(args.exclude)
    combinations = parse_combine_spec(args.combine)

    # Load and process data
    try:
        headers, rows = parse_csv(args.csv_file)
    except FileNotFoundError:
        print(f"Error: File '{args.csv_file}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate columns
    all_cols = set(x_cols + (legend_cols or []) + [value_col])
    missing_cols = all_cols - set(headers)
    if missing_cols:
        print(
            f"Error: Columns not found in CSV: {', '.join(missing_cols)}",
            file=sys.stderr,
        )
        print(f"Available columns: {', '.join(headers)}", file=sys.stderr)
        sys.exit(1)

    # Aggregate and process data
    data = aggregate_data(rows, x_cols, legend_cols, value_col)

    # Apply exclusions
    data = apply_exclusions(data, exclude_list)

    # Apply combinations (returns data and map of which series used special aggregation)
    data, combination_agg_map = apply_combinations(data, combinations)

    # Aggregate values (sum is default for backward compatibility)
    # Note: combined series have already been aggregated by their specified method
    data = aggregate_values(data, agg_method="sum")

    if not data:
        print("No data to plot", file=sys.stderr)
        sys.exit(1)

    if args.vertical:
        plot_bars_vertical(
            data,
            x_cols,
            legend_cols,
            value_col,
            custom_colors,
            custom_patterns,
            renames,
            bar_height=args.height,
            show_values=not args.no_values,
        )
    else:
        plot_bars(
            data,
            x_cols,
            legend_cols,
            value_col,
            custom_colors,
            custom_patterns,
            renames,
            bar_width=args.width,
            show_values=not args.no_values,
        )


if __name__ == "__main__":
    main()
