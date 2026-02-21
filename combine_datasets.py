"""
Combine Multiple Year Datasets

Combines multiple CSV files (e.g., 2023_data.csv, 2024_data.csv) into a single
dataset with a 'year' column. If the year column doesn't exist, it extracts
the year from the filename.

Usage:
    python combine_datasets.py
    python combine_datasets.py --files 2023_data.csv 2024_data.csv 2025_data.csv
    python combine_datasets.py --files *_data.csv --output combined_data.csv
"""

import argparse
import glob
import re
from pathlib import Path

import pandas as pd


def extract_year_from_filename(filename: str) -> int | None:
    """
    Extract a 4-digit year from a filename.

    Args:
        filename: Name of the file (e.g., "2023_data.csv")

    Returns:
        Year as integer, or None if not found
    """
    match = re.search(r"\b(20\d{2})\b", filename)
    return int(match.group(1)) if match else None


def combine_datasets(
    files: list[str],
    output_file: str = "combined_data.csv",
    year_column: str = "year",
) -> None:
    """
    Combine multiple CSV files into one with a year column.

    Args:
        files: List of CSV file paths to combine
        output_file: Path for the combined output file
        year_column: Name of the year column to add/use
    """
    dataframes = []

    for file in files:
        file_path = Path(file)
        if not file_path.exists():
            print(f"Warning: File not found: {file}")
            continue

        print(f"Reading: {file}")
        df = pd.read_csv(file)

        # Add year column if it doesn't exist
        if year_column not in df.columns:
            year = extract_year_from_filename(file)
            if year:
                df[year_column] = year
                print(f"  Added year column: {year}")
            else:
                print(f"  Warning: Could not extract year from filename")
        else:
            print(f"  Using existing year column")

        dataframes.append(df)
        print(f"  Shape: {df.shape}")

    if not dataframes:
        print("Error: No valid dataframes to combine")
        return

    # Combine all dataframes
    print("\nCombining dataframes...")
    combined = pd.concat(dataframes, ignore_index=True)
    print(f"Combined shape: {combined.shape}")

    # Save to output file
    combined.to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")

    # Print year distribution
    if year_column in combined.columns:
        print(f"\nYear distribution:")
        print(combined[year_column].value_counts().sort_index())


def expand_glob_patterns(files: list[str]) -> list[str]:
    """
    Expand glob patterns in file list.

    Args:
        files: List of file paths, possibly containing glob patterns

    Returns:
        Expanded list of file paths
    """
    expanded = []
    for pattern in files:
        # Try to expand as glob pattern
        matches = glob.glob(pattern)
        if matches:
            expanded.extend(sorted(matches))
        else:
            # No matches, keep original (might be a literal filename)
            expanded.append(pattern)
    return expanded


def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple FRC match dataset CSV files into one"
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=["*_data.csv"],
        help="CSV files/patterns to combine (default: *_data.csv)",
    )
    parser.add_argument(
        "--output",
        default="combined_data.csv",
        help="Output filename (default: combined_data.csv)",
    )
    parser.add_argument(
        "--year-column",
        default="year",
        help="Name of the year column (default: year)",
    )

    args = parser.parse_args()

    # Expand glob patterns
    expanded_files = expand_glob_patterns(args.files)
    print(f"Found {len(expanded_files)} files: {expanded_files}\n")

    combine_datasets(
        files=expanded_files,
        output_file=args.output,
        year_column=args.year_column,
    )


if __name__ == "__main__":
    main()
