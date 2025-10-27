"""
Entry point for command line interface.
"""

import argparse
import os
import pathlib
import sys
import time
import traceback
from importlib.metadata import version

import pandas as pd

from dca.adca.adca import process_file
from dca.adca.cmd_init import init_yaml_and_csv


def subcommand_init(args):
    """Execute the command `adca init`."""
    init_yaml_and_csv(filename="demo")


def subcommand_run(args):
    """Execute the command `adca run`."""
    assert args.hyperparam_maxfun > 0
    assert args.plot_verbosity >= 0

    # Convert everything to paths
    filenames = list(map(pathlib.Path, args.filenames))

    # .yaml files to process
    process = [p for p in filenames if p.is_file() and p.suffix == ".yaml"]

    # Extend with .yaml files in directories
    for path in filenames:
        if path.is_file():
            continue
        process.extend(list(path.rglob("*.yaml")))

    # List files to process
    print("Files to process:")
    for i, yaml_file in enumerate(process, 1):
        print(f"  ({i}): {yaml_file}")

    time.sleep(3)

    # Process each file
    CURRENT_DIRECTORY = pathlib.Path(os.getcwd())
    NOW = pd.Timestamp.now().strftime("%Y-%m-%d-%H-%M")
    for i, yaml_file in enumerate(process, 1):
        print(f"  ({i}): Processing {yaml_file}")
        try:
            process_file(
                config_path=yaml_file,
                current_directory=CURRENT_DIRECTORY,
                current_time=NOW,
                hyperparam_maxfun=args.hyperparam_maxfun,
                plot_verbosity=args.plot_verbosity,
            )
        except Exception:
            traceback.print_exc()  # Print the exception
            print(
                f"""
ADCA {version("dca")} has raised an exception on file {yaml_file}.
If you are unable to fix the issue, then contact us for help.
You may use the public issue tracker if you are not in Equinor.

The issue tracker is PUBLIC, so do not upload ANY sensitive information.

- Documentation:        https://dsadocs.equinor.com/docs/decline-curve-analysis/
- Public issue tracker: https://github.com/equinor/decline-curve-analysis/issues
- Help (Equinor):       Contact Tommy Odland (todl) or Knut Utne Hollund (kuho)."""
            )

            sys.exit(1)  # Exit with non-zero code

    print("Finished processing all files.")


def run():
    """Main entry point."""

    # Set up argument parses and parse arguments
    parser = argparse.ArgumentParser(
        prog="ADCA - Automatic Decline Curve Analysis",
        description="""For more info: https://dsadocs.equinor.com/docs/decline-curve-analysis/index.html""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"ADCA {version('dca')}"
    )

    subparsers = parser.add_subparsers(dest="subparser_name")

    # the RUN subcommand
    description = "Run ADCA on a .yaml config file."
    parser_run = subparsers.add_parser("run", help=description, description=description)
    parser_run.add_argument(
        "filenames", type=str, nargs="+", help="Either a file or a directory."
    )
    parser_run.add_argument(
        "-hm",
        "--hyperparam-maxfun",
        dest="hyperparam_maxfun",
        type=int,
        nargs="?",
        help="Function calls (iterations) to hyperparameter optimization.",
        default=25,
    )
    parser_run.add_argument(
        "-pv",
        "--plot-verbosity",
        dest="plot_verbosity",
        type=int,
        nargs="?",
        help="Integer >= 0 indicating how many types of debugging plots (.png files) to produce.",
        default=1,
    )
    parser_run.set_defaults(func=subcommand_run)

    # the INIT subcommand
    description = "Initialize demo .yaml and .csv files."
    parser_init = subparsers.add_parser(
        "init", help=description, description=description
    )
    parser_init.set_defaults(func=subcommand_init)

    # missing arguments, print help and exit
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    # Backwards compatibility. If not a known command, assume "run"
    if (sys.argv[1] not in subparsers.choices) and (not sys.argv[1].startswith("-")):
        sys.argv.insert(1, "run")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    run()
