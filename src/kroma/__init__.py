"""KROMA: ontology matching with knowledge retrieval and large language models.

KROMA decides whether a concept in a source ontology and a concept in a target
ontology refer to the same real-world entity. For each candidate pair it
retrieves the concepts' context (ontology structure, cached dictionaries,
SPARQL endpoints and ConceptNet), asks one or more language models for a
Yes/No answer with a confidence, and refines a bisimulation-based partition of
the merged ontology graph as answers arrive.

The usual entry points are the ``kroma`` command (:mod:`kroma.cli`) and
:class:`kroma.pipeline.MatchingRun`.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
