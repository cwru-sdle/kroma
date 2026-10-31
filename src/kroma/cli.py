"""Command line for KROMA experiments.

The ``kroma`` console script calls :func:`main`.

* ``kroma match SOURCE TARGET -o OUT`` aligns two ontology files with
  :func:`kroma.match`.
* ``kroma run`` and the bare flag form (``kroma --method_config ... --llm
  ...``) start one benchmark :class:`~kroma.pipeline.MatchingRun`.
"""

import argparse
import importlib.util
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


def _add_match_args(parser: argparse.ArgumentParser) -> None:
    from kroma.matcher import DEFAULT_BACKEND, DEFAULT_MODEL

    parser.add_argument("source", help="Source ontology file or URL (.owl, .ttl, ...)")
    parser.add_argument("target", help="Target ontology file or URL")
    parser.add_argument("-o", "--output", default="alignments.csv",
                        help="Output file: .csv, .tsv or .rdf (OAEI format). Default: alignments.csv")
    parser.add_argument("--backend", default=DEFAULT_BACKEND, help=f"Chat backend (default: {DEFAULT_BACKEND})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Chat model (default: {DEFAULT_MODEL})")
    parser.add_argument("--top-k", type=int, default=3, help="Candidates per source class (default: 3)")
    parser.add_argument("--min-similarity", type=float, default=0.01,
                        help="Skip candidates below this retrieval similarity (default: 0.01)")
    parser.add_argument("--min-confidence", type=int, default=0, help="Minimum model confidence, 0-10")
    parser.add_argument("--all-matches", action="store_true",
                        help="Keep every accepted candidate, not just the first per source class")
    parser.add_argument("--no-exact", action="store_true",
                        help="Ask the model even when names match exactly")
    parser.add_argument("--embedder", default="lexical",
                        help='"lexical" (default) or e.g. "huggingface:allenai/scibert_scivocab_uncased"')
    parser.add_argument("--workers", type=int, default=4, help="Parallel model requests (default: 4)")
    parser.add_argument("--limit", type=int, help="Only match the first N source classes")


def match_command(args) -> None:
    """Run :func:`kroma.match` for ``kroma match`` and save the alignments."""
    import logging
    import os

    from kroma.matcher import match
    from kroma.utils import env  # noqa: F401  (loads .env)

    key = f"{args.backend.upper()}_API_KEY"
    if not os.environ.get(key):
        raise SystemExit(f"Set {key} (in the environment or a .env file) to use the {args.backend} backend.")
    client = {"openai": "openai", "togetherai": "together"}.get(args.backend)
    if client and importlib.util.find_spec(client) is None:
        raise SystemExit(f"The {args.backend} backend needs its client: pip install {client}")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = match(
        args.source,
        args.target,
        backend=args.backend,
        model=args.model,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
        min_confidence=args.min_confidence,
        exact_match=not args.no_exact,
        first_match_only=not args.all_matches,
        embedder=args.embedder,
        workers=args.workers,
        limit=args.limit,
    )
    path = result.save(args.output)
    print(
        f"{len(result)} alignments ({result.exact_matches} exact) written to {path}; "
        f"{result.pairs_judged} pairs judged in {result.llm_calls} model calls, "
        f"{result.input_tokens + result.output_tokens} tokens"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``kroma`` parser with its ``match`` and ``run`` subcommands.

    Returns:
        argparse.ArgumentParser: Parser whose ``command`` attribute names the
        subcommand used.
    """
    parser = argparse.ArgumentParser(
        prog="kroma",
        description="KROMA ontology matching with knowledge retrieval and LLMs",
    )
    sub = parser.add_subparsers(dest="command")
    _add_match_args(sub.add_parser("match", help="Align two ontology files"))
    run = sub.add_parser("run", help="Run a benchmark experiment from experiments/configs")
    _add_run_args(run)
    return parser


def build_run_parser() -> argparse.ArgumentParser:
    """Build the parser for the flag-only form of a run.

    This accepts the same flags as ``kroma run`` without the subcommand.

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
    if argv[0] == "match":
        match_command(build_parser().parse_args(argv))
        return 0
    if argv[0] == "run":
        args = build_parser().parse_args(argv)
    else:
        args = build_run_parser().parse_args(argv)
    RunCommand(args).execute()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
