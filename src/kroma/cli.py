"""Command line for KROMA experiments.

The ``kroma`` console script calls :func:`main`. ``kroma run`` and the bare
flag form (``kroma --method_config ... --llm ...``) both start one
:class:`~kroma.pipeline.MatchingRun`.
"""

import argparse
import sys

from kroma.config.constants import EVAL_SIZE_FRACTIONS


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method_config", required=True)
    parser.add_argument("--llm", required=True)
    parser.add_argument("--size", choices=list(EVAL_SIZE_FRACTIONS), default="full")
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--active_learning", action="store_true")
    parser.add_argument("--compare_models", action="store_true")
    parser.add_argument("--bisim", action="store_true")
    parser.add_argument("--debate", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``kroma`` parser with its ``run`` subcommand.

    Returns:
        argparse.ArgumentParser: Parser whose ``command`` attribute is
        ``"run"`` when the subcommand is used.
    """
    parser = argparse.ArgumentParser(
        prog="kroma",
        description="Evaluate KROMA ontology matching",
    )
    sub = parser.add_subparsers(dest="command")
    run = sub.add_parser("run", help="Run an ontology matching experiment")
    _add_run_args(run)
    return parser


def build_run_parser() -> argparse.ArgumentParser:
    """Build the parser for the flag-only form of a run.

    This accepts the same flags as ``kroma run`` without the subcommand, as
    the original ``python -m main`` script did.

    Returns:
        argparse.ArgumentParser: Parser for the run flags.
    """
    parser = argparse.ArgumentParser(
        prog="kroma",
        description="Evaluate KROMA ontology matching",
    )
    _add_run_args(parser)
    return parser


class RunCommand:
    """Command that executes one matching experiment.

    Args:
        args (argparse.Namespace): Parsed run flags, as produced by
            :func:`build_parser` or :func:`build_run_parser`.
    """

    def __init__(self, args):
        self.args = args

    def execute(self) -> None:
        """Run the experiment.

        The pipeline is imported lazily so ``kroma --help`` works without the
        optional model dependencies.

        Raises:
            SystemExit: If a model dependency is missing; the message names
                the package and how to install the ``models`` extra.
        """
        try:
            from kroma.pipeline import MatchingRun
        except ModuleNotFoundError as exc:
            missing = exc.name or "a package"
            raise SystemExit(
                f"Missing dependency '{missing}'. "
                "Install model backends with: uv pip install '.[models]'"
            ) from exc
        MatchingRun(self.args).execute()


def main(argv=None) -> int:
    """Entry point of the ``kroma`` console script.

    Args:
        argv (list[str] | None): Arguments without the program name. Defaults
            to ``sys.argv[1:]``.

    Returns:
        int: Process exit code, ``0`` on success.
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help"):
        build_parser().print_help()
        return 0
    if argv[0] == "run":
        args = build_parser().parse_args(argv)
    else:
        args = build_run_parser().parse_args(argv)
    RunCommand(args).execute()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
