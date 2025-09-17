#!/usr/bin/env python3
"""
Script to combine kernel CSV files across multiple folders,
keeping only speedup columns and removing average rows.
Also extracts best configs from log files and adds them as a column.
Folders are defined within the script.
"""

import pandas as pd
import sys
from pathlib import Path
import time
from datetime import datetime
import re


# Define kernel names
KERNEL_NAME_LIST = [
    "rms_norm",
    "layer_norm",
    "softmax",
    "cross_entropy",
    "sum",
    "jagged_mean",
    "vector_add",
    "embedding",
    "vector_exp",
]

# Define folders to process
# You can use glob patterns or explicit folder names
FOLDER_LIST = [
    "benchmarks_results/benchmarks_autotune_1756925459_input_shard_1_of_4",
    "benchmarks_results/benchmarks_autotune_1756925461_input_shard_2_of_4",
    "benchmarks_results/benchmarks_autotune_1756925463_input_shard_3_of_4",
    "benchmarks_results/benchmarks_autotune_1756925466_input_shard_4_of_4",
]


def expand_folder_patterns(folder_patterns):
    """
    Expand glob patterns in folder list to actual folder paths.
    
    Args:
        folder_patterns: List of folder paths or glob patterns
    
    Returns:
        List of actual folder paths
    """
    expanded_folders = []
    
    for pattern in folder_patterns:
        path = Path(pattern)
        
        # Check if it's a glob pattern (contains * or ?)
        if '*' in pattern or '?' in pattern:
            # Expand glob pattern
            matching_paths = list(Path('.').glob(pattern))
            # Filter to only directories
            matching_dirs = [p for p in matching_paths if p.is_dir()]
            expanded_folders.extend(matching_dirs)
        else:
            # Direct path - check if it exists and is a directory
            if path.exists() and path.is_dir():
                expanded_folders.append(path)
            else:
                print(f"Warning: '{pattern}' is not a valid directory")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_folders = []
    for folder in expanded_folders:
        folder_abs = folder.resolve()
        if folder_abs not in seen:
            seen.add(folder_abs)
            unique_folders.append(folder)
    
    return unique_folders


def extract_configs_from_log(log_file):
    """
    Extract all helion config occurrences from a log file.
    
    Args:
        log_file: Path to the log file
    
    Returns:
        List of config strings in order of appearance
    """
    configs = []
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Pattern to match the config lines
        # Look for lines that start with '@helion.kernel(config=' after the "One can hardcode" line
        pattern = r'One can hardcode the best config and skip autotuning with:\s*\n\s*(@helion\.kernel\(config=helion\.Config\([^)]+\)\))'
        
        matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            # Clean up the config string - remove extra whitespace
            config = match.strip()
            configs.append(config)
        
        print(f"    Extracted {len(configs)} configs from log file")
        
    except Exception as e:
        print(f"    Error reading log file {log_file}: {e}")
    
    return configs


def get_helion_column_name(df):
    """
    Find the helion column name pattern in the dataframe.
    
    Args:
        df: DataFrame with columns
    
    Returns:
        Base name for helion config column (e.g., 'helion_attention' -> 'helion_attention-config')
    """
    columns = df.columns.tolist()
    
    # Look for any column that starts with 'helion' and contains 'speedup'
    for col in columns:
        if col.startswith('helion') and 'speedup' in col:
            # Extract the base name (everything before '-speedup')
            base_name = col.replace('-speedup', '')
            return f"{base_name}-config"
    
    # Fallback to generic name if no helion speedup column found
    for col in columns:
        if col.startswith('helion'):
            # Extract the base name (everything before the last '-')
            parts = col.rsplit('-', 1)
            if len(parts) > 1:
                return f"{parts[0]}-config"
    
    # Default fallback
    return "helion-config"


def combine_kernel_csvs(kernel_name, folder_list, output_dir):
    """
    Combine CSV files for a specific kernel across multiple folders.
    Also extracts configs from corresponding log files.
    
    Args:
        kernel_name: Name of the kernel to process
        folder_list: List of folders to search in
        output_dir: Directory to save combined CSV
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\nProcessing kernel: {kernel_name}")
    print("-" * 50)
    
    dataframes = []
    all_configs = []
    found_count = 0
    
    # Search for kernel CSV and log files in each folder
    for folder in folder_list:
        csv_file = folder / f"{kernel_name}_cached.csv"
        log_file = folder / f"{kernel_name}_cached.log"
        
        if csv_file.exists():
            print(f"  Found: {csv_file}")
            try:
                # Read CSV with semicolon delimiter
                df = pd.read_csv(csv_file, delimiter=';')
                
                if not df.empty:
                    # Extract configs from log file if it exists
                    folder_configs = []
                    if log_file.exists():
                        print(f"  Found log: {log_file}")
                        folder_configs = extract_configs_from_log(log_file)
                    
                    # Filter out average rows BEFORE matching configs
                    first_col = df.columns[0]
                    first_col_lower = df[first_col].astype(str).str.lower()
                    mask = ~first_col_lower.isin(['average'])
                    df_filtered = df[mask].copy()
                    
                    # Match configs to non-average rows
                    if folder_configs:
                        if len(folder_configs) == len(df_filtered):
                            # Add configs for this folder's rows
                            all_configs.extend(folder_configs)
                        else:
                            raise Exception(f"Config count ({len(folder_configs)}) doesn't match non-average row count ({len(df_filtered)})")
                    else:
                        # No configs found, add empty strings
                        all_configs.extend([''] * len(df_filtered))
                    
                    dataframes.append(df_filtered)
                    found_count += 1
                else:
                    print(f"    Warning: Empty CSV file")
                    
            except Exception as e:
                print(f"    Error reading {csv_file}: {e}")
    
    if not dataframes:
        print(f"  No valid CSV files found for kernel '{kernel_name}'")
        return False
    
    print(f"  Found {found_count} valid CSV file(s)")
    
    try:
        # Merge all dataframes (concatenate rows)
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"  Combined shape: {combined_df.shape}")
        
        # Get column names
        columns = combined_df.columns.tolist()
        if not columns:
            print("  Error: No columns found")
            return False
        
        # Keep first column and columns containing "speedup"
        first_col = columns[0]
        speedup_cols = [col for col in columns[1:] if 'speedup' in col.lower()]
        
        # Create list of columns to keep
        cols_to_keep = [first_col] + speedup_cols
        
        print(f"  Keeping {len(cols_to_keep)} columns: {first_col} + {len(speedup_cols)} speedup columns")
        
        # Filter columns
        filtered_df = combined_df[cols_to_keep]
        
        # Add config column if we have configs
        if all_configs and len(all_configs) == len(filtered_df):
            config_col_name = get_helion_column_name(combined_df)
            filtered_df[config_col_name] = all_configs
            print(f"  Added config column: {config_col_name}")
        elif all_configs:
            raise Exception(f"Total config count ({len(all_configs)}) doesn't match final row count ({len(filtered_df)})")
        
        print(f"  Final shape: {filtered_df.shape}")
        
        # Save to output file
        output_file = output_dir / f"{kernel_name}_combined.csv"
        filtered_df.to_csv(output_file, sep=';', index=False)
        print(f"  ✓ Saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"  Error processing kernel '{kernel_name}': {e}")
        return False


def main():
    """Main function to process kernels across predefined folders."""
    
    print(f"CSV Kernel Combiner with Config Extraction")
    print(f"=" * 70)
    
    # Expand folder patterns to actual paths
    print("Expanding folder patterns...")
    folders = expand_folder_patterns(FOLDER_LIST)
    
    if not folders:
        print("\nError: No valid folders found!")
        print("Please check FOLDER_LIST in the script.")
        print(f"Current patterns: {FOLDER_LIST}")
        sys.exit(1)
    
    print(f"\nFound {len(folders)} folder(s):")
    for folder in folders[:5]:  # Show first 5 folders
        print(f"  - {folder}")
    if len(folders) > 5:
        print(f"  ... and {len(folders) - 5} more")
    
    # Create output directory with timestamp
    timestamp = int(time.time())
    output_dir = Path(f"benchmarks_results/benchmarks_autotune_{timestamp}_input_combined")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Processing {len(KERNEL_NAME_LIST)} kernels...")
    
    # Process each kernel
    successful = 0
    failed = 0
    
    for kernel_name in KERNEL_NAME_LIST:
        if combine_kernel_csvs(kernel_name, folders, output_dir):
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"Summary: Successfully processed {successful} kernels, {failed} failed")
    print(f"Output saved to: {output_dir}")
    
    # List created files
    created_files = list(output_dir.glob("*_combined.csv"))
    if created_files:
        print(f"\nCreated files:")
        for f in sorted(created_files):
            size_kb = f.stat().st_size / 1024
            print(f"  - {f.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    # Check if any command line arguments were provided
    if len(sys.argv) > 1:
        print("Note: This script uses predefined folders from FOLDER_LIST.")
        print("Command line arguments are ignored.")
        print("To use different folders, edit FOLDER_LIST in the script.\n")
    
    main()
