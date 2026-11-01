"""KROMA: ontology matching with knowledge retrieval and large language models.

KROMA decides whether a concept in a source ontology and a concept in a target
ontology refer to the same real-world entity. For each candidate pair it
retrieves the concepts' context (ontology structure, cached dictionaries,
SPARQL endpoints and ConceptNet), asks one or more language models for a
Yes/No answer with a confidence, and refines a bisimulation-based partition of
the merged ontology graph as answers arrive.

To align two ontologies, call :func:`kroma.match` (or run ``kroma match``).
To reproduce the paper's benchmarks, use ``kroma run`` and
:class:`kroma.pipeline.MatchingRun`.

Example:
    >>> import kroma
    >>> result = kroma.match("source.owl", "target.owl")  # doctest: +SKIP
    >>> result.save("alignments.csv")  # doctest: +SKIP
"""

__version__ = "0.2.0"

from kroma.alignment import Alignment, MatchResult, save_alignments
from kroma.matcher import find_candidates, match
from kroma.ontology import Concept, Ontology, load_ontology

__all__ = [
    "__version__",
    "match",
    "find_candidates",
    "load_ontology",
    "Ontology",
    "Concept",
    "Alignment",
    "MatchResult",
    "save_alignments",
]
