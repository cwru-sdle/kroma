"""Alignments produced by :func:`kroma.match`, and writers for them."""

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, List, Union
from xml.sax.saxutils import escape

FIELDS = ["source", "target", "source_label", "target_label", "relation",
          "confidence", "similarity", "method"]


@dataclass
class Alignment:
    """One correspondence between a source and a target concept.

    Attributes:
        source (str): Source class IRI.
        target (str): Target class IRI.
        source_label (str): Source preferred label.
        target_label (str): Target preferred label.
        confidence (int): Model confidence from 0 to 10 (10 for exact matches).
        similarity (float): Retrieval similarity of the pair, -1 to 1.
        method (str): ``"exact"`` when a label or synonym matched exactly,
            ``"llm"`` when the language model decided.
        relation (str): Always ``"="`` (equivalence).
    """

    source: str
    target: str
    source_label: str
    target_label: str
    confidence: int
    similarity: float
    method: str
    relation: str = "="

    def as_dict(self) -> dict:
        """Return the alignment as a plain dict, in :data:`FIELDS` order."""
        row = asdict(self)
        return {key: row[key] for key in FIELDS}


@dataclass
class MatchResult:
    """The alignments from one :func:`kroma.match` call and what they cost.

    Iterating over a result yields its :class:`Alignment` objects.

    Attributes:
        alignments (list[Alignment]): Accepted alignments, in source order.
        pairs_judged (int): Candidate pairs sent to the language model.
        exact_matches (int): Alignments accepted without the model.
        llm_calls (int): Model requests made.
        input_tokens (int): Prompt tokens, when the model reports them.
        output_tokens (int): Completion tokens, when the model reports them.
    """

    alignments: List[Alignment] = field(default_factory=list)
    pairs_judged: int = 0
    exact_matches: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def __iter__(self) -> Iterator[Alignment]:
        return iter(self.alignments)

    def __len__(self) -> int:
        return len(self.alignments)

    def __getitem__(self, index):
        return self.alignments[index]

    def save(self, path: Union[str, Path]) -> Path:
        """Write the alignments; see :func:`save_alignments`."""
        return save_alignments(self.alignments, path)


def save_alignments(alignments, path: Union[str, Path]) -> Path:
    """Write alignments to CSV, TSV or the OAEI Alignment format.

    The format follows the suffix: ``.csv``, ``.tsv``, or ``.rdf``/``.xml``
    for the `Alignment API <https://moex.gitlabpages.inria.fr/alignapi/format.html>`_
    RDF format used by OAEI, with confidence scaled to 0-1.

    Args:
        alignments (Iterable[Alignment]): Alignments to write.
        path (str | pathlib.Path): Output file.

    Returns:
        pathlib.Path: The written path.

    Raises:
        ValueError: If the suffix is not supported.
    """
    path = Path(path)
    alignments = list(alignments)
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, FIELDS, delimiter="," if suffix == ".csv" else "\t")
            writer.writeheader()
            writer.writerows(a.as_dict() for a in alignments)
    elif suffix in (".rdf", ".xml"):
        path.write_text(_alignment_rdf(alignments), encoding="utf-8")
    else:
        raise ValueError(f"unsupported alignment format {suffix!r}; use .csv, .tsv or .rdf")
    return path


def _alignment_rdf(alignments: List[Alignment]) -> str:
    cells = "".join(
        "  <map>\n    <Cell>\n"
        f'      <entity1 rdf:resource="{escape(a.source, {chr(34): "&quot;"})}"/>\n'
        f'      <entity2 rdf:resource="{escape(a.target, {chr(34): "&quot;"})}"/>\n'
        f'      <relation>{escape(a.relation)}</relation>\n'
        f'      <measure rdf:datatype="http://www.w3.org/2001/XMLSchema#float">{a.confidence / 10:.2f}</measure>\n'
        "    </Cell>\n  </map>\n"
        for a in alignments
    )
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<rdf:RDF xmlns="http://knowledgeweb.semanticweb.org/heterogeneity/alignment"\n'
        '         xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        "<Alignment>\n  <xml>yes</xml>\n  <level>0</level>\n  <type>??</type>\n"
        f"{cells}</Alignment>\n</rdf:RDF>\n"
    )
