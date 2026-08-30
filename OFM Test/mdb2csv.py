#!/usr/bin/env python3
"""
Script to export all tables from an Access MDB file to CSV using mdb-tools.

Requirements:
- mdb-tools installed (brew install mdb-tools on macOS)
- Python 3.6+

Usage:
    python export_mdb_to_csv.py input.mdb [output_directory]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def get_table_list(mdb_file):
    """
    Get list of tables from MDB file using mdb-tables command.

    Parameters
    ----------
    mdb_file : str
        Path to the MDB file

    Returns
    -------
    list
        List of table names

    Raises
    ------
    subprocess.CalledProcessError
        If mdb-tables command fails
    """
    try:
        result = subprocess.run(
            ["mdb-tables", "-1", mdb_file], capture_output=True, text=True, check=True
        )
        # Filter out empty lines and strip whitespace
        tables = [table.strip() for table in result.stdout.split("\n") if table.strip()]
        return tables
    except subprocess.CalledProcessError as e:
        print(f"Error running mdb-tables: {e}")
        print(f"stderr: {e.stderr}")
        raise


def export_table_to_csv(mdb_file, table_name, output_dir):
    """
    Export a single table to CSV using mdb-export command.

    Parameters
    ----------
    mdb_file : str
        Path to the MDB file
    table_name : str
        Name of the table to export
    output_dir : str
        Directory to save the CSV file

    Returns
    -------
    bool
        True if export successful, False otherwise
    """
    output_file = Path(output_dir) / f"{table_name}.csv"

    try:
        with open(output_file, "w") as f:
            subprocess.run(["mdb-export", mdb_file, table_name], stdout=f, check=True)
        print(f"✓ Exported {table_name} to {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to export {table_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Export all tables from Access MDB file to CSV files"
    )
    parser.add_argument("mdb_file", help="Path to the MDB file")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="./csv_output",
        help="Output directory for CSV files (default: ./csv_output)",
    )
    parser.add_argument(
        "--list-only", action="store_true", help="Only list tables, do not export"
    )

    args = parser.parse_args()

    # Check if mdb file exists
    if not os.path.exists(args.mdb_file):
        print(f"Error: MDB file '{args.mdb_file}' not found")
        sys.exit(1)

    # Check if mdb-tools are installed
    try:
        subprocess.run(["mdb-tables", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: mdb-tools not found. Install with: brew install mdb-tools")
        sys.exit(1)

    # Get list of tables
    print(f"Reading tables from {args.mdb_file}...")
    try:
        tables = get_table_list(args.mdb_file)
    except subprocess.CalledProcessError:
        sys.exit(1)

    if not tables:
        print("No tables found in the MDB file")
        sys.exit(0)

    print(f"Found {len(tables)} tables:")
    for i, table in enumerate(tables, 1):
        print(f"  {i:2d}. {table}")

    if args.list_only:
        sys.exit(0)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"\nExporting to directory: {output_dir}")

    # Export each table
    successful_exports = 0
    for table in tables:
        if export_table_to_csv(args.mdb_file, table, output_dir):
            successful_exports += 1

    print(
        f"\nExport complete: {successful_exports}/{len(tables)} tables exported successfully"
    )


if __name__ == "__main__":
    main()
