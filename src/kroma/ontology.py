"""Load an ontology into the concepts KROMA matches.

:func:`load_ontology` reads any RDF serialisation rdflib understands (RDF/XML
``.owl``, Turtle, N-Triples, JSON-LD, ...) and keeps, for every named class,
its labels, synonyms, parents and children.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS
from rdflib.util import guess_format

OBO_IN_OWL = Namespace("http://www.geneontology.org/formats/oboInOwl#")

LABEL_PREDICATES = (RDFS.label, SKOS.prefLabel)
SYNONYM_PREDICATES = (
    SKOS.altLabel,
    OBO_IN_OWL.hasExactSynonym,
    OBO_IN_OWL.hasRelatedSynonym,
    OBO_IN_OWL.hasNarrowSynonym,
    OBO_IN_OWL.hasBroadSynonym,
)

_SUFFIX_FORMATS = {".owl": "xml", ".rdf": "xml", ".xml": "xml", ".obo": None}


@dataclass
class Concept:
    """A named class of an ontology.

    Attributes:
        uri (str): The class IRI.
        label (str): Preferred label: the first ``rdfs:label`` or
            ``skos:prefLabel`` (English preferred), else a name derived from
            the IRI.
        labels (list[str]): All labels, preferred first.
        synonyms (list[str]): ``skos:altLabel`` and oboInOwl synonyms.
        parents (list[str]): IRIs of named superclasses.
        children (list[str]): IRIs of named subclasses.
    """

    uri: str
    label: str
    labels: List[str] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)

    @property
    def names(self) -> List[str]:
        """list[str]: Labels followed by synonyms, without duplicates."""
        return list(dict.fromkeys(self.labels + self.synonyms))


class Ontology:
    """The classes of one ontology, keyed by IRI.

    Build one with :func:`load_ontology` or :meth:`from_graph`. It behaves like
    a read-only mapping from IRI to :class:`Concept`.

    Args:
        concepts (dict[str, Concept]): Concepts keyed by IRI.
        name (str): Display name, such as the file name.
    """

    def __init__(self, concepts: Dict[str, Concept], name: str = "ontology"):
        self.concepts = concepts
        self.name = name

    def __len__(self) -> int:
        return len(self.concepts)

    def __iter__(self) -> Iterator[Concept]:
        return iter(self.concepts.values())

    def __getitem__(self, uri: str) -> Concept:
        return self.concepts[uri]

    def __contains__(self, uri: object) -> bool:
        return uri in self.concepts

    def __repr__(self) -> str:
        return f"Ontology({self.name!r}, {len(self)} classes)"

    def labels_of(self, uris: List[str]) -> List[str]:
        """Return the preferred labels of the given IRIs that are in this ontology."""
        return [self.concepts[u].label for u in uris if u in self.concepts]

    @classmethod
    def from_graph(cls, graph: Graph, name: str = "ontology") -> "Ontology":
        """Collect the named, non-deprecated classes of an rdflib graph.

        Classes are subjects typed ``owl:Class`` or ``rdfs:Class``, plus any
        IRI used in ``rdfs:subClassOf``. ``owl:Thing`` is left out.

        Args:
            graph (rdflib.Graph): Parsed ontology.
            name (str): Display name.

        Returns:
            Ontology: The ontology's concepts.
        """
        uris = set()
        for cls_type in (OWL.Class, RDFS.Class):
            uris.update(s for s in graph.subjects(RDF.type, cls_type) if isinstance(s, URIRef))
        for s, o in graph.subject_objects(RDFS.subClassOf):
            if isinstance(s, URIRef):
                uris.add(s)
            if isinstance(o, URIRef):
                uris.add(o)
        uris.discard(OWL.Thing)
        uris = {u for u in uris if not _is_true(graph.value(u, OWL.deprecated))}

        concepts: Dict[str, Concept] = {}
        for uri in sorted(uris, key=str):
            labels = _literals(graph, uri, LABEL_PREDICATES)
            if not labels:
                labels = [local_name(str(uri))]
            synonyms = [s for s in _literals(graph, uri, SYNONYM_PREDICATES) if s not in labels]
            parents = sorted(
                str(o) for o in graph.objects(uri, RDFS.subClassOf)
                if isinstance(o, URIRef) and o in uris
            )
            concepts[str(uri)] = Concept(str(uri), labels[0], labels, synonyms, parents)
        for concept in concepts.values():
            for parent in concept.parents:
                concepts[parent].children.append(concept.uri)
        return cls(concepts, name)


def load_ontology(
    source: Union[str, Path, Graph, Ontology],
    format: Optional[str] = None,
) -> Ontology:
    """Load an ontology from a file, URL or rdflib graph.

    Args:
        source (str | pathlib.Path | rdflib.Graph | Ontology): Path or URL of an
            RDF file, a parsed graph, or an :class:`Ontology` (returned as is).
        format (str | None): rdflib format name such as ``"xml"`` or
            ``"turtle"``. Guessed from the file suffix when omitted; ``.owl``
            files are read as RDF/XML.

    Returns:
        Ontology: The ontology's concepts.

    Example:
        >>> onto = load_ontology("mouse.owl")  # doctest: +SKIP
        >>> onto["http://mouse.owl#MA_0000072"].label  # doctest: +SKIP
        'heart'
    """
    if isinstance(source, Ontology):
        return source
    if isinstance(source, Graph):
        return Ontology.from_graph(source)
    location = str(source)
    if format is None:
        suffix = Path(location.split("?")[0]).suffix.lower()
        format = _SUFFIX_FORMATS.get(suffix) or guess_format(location) or "xml"
    graph = Graph()
    graph.parse(location, format=format)
    return Ontology.from_graph(graph, name=Path(location).name or location)


def local_name(uri: str) -> str:
    """Turn the last segment of an IRI into words.

    Example:
        >>> local_name("http://example.org/onto#HeartValve")
        'heart valve'
        >>> local_name("http://purl.obolibrary.org/obo/blood_vessel")
        'blood vessel'
    """
    name = re.split(r"[#/]", uri.rstrip("/#"))[-1]
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return re.sub(r"[_\-]+", " ", name).strip().lower()


def _literals(graph: Graph, uri: URIRef, predicates) -> List[str]:
    """Literal values of ``predicates``, English or untagged first, deduplicated."""
    found = []
    for predicate in predicates:
        for obj in graph.objects(uri, predicate):
            if isinstance(obj, Literal) and str(obj).strip():
                found.append((obj.language not in (None, "en"), str(obj).strip()))
    found.sort(key=lambda item: item[0])
    return list(dict.fromkeys(text for _, text in found))


def _is_true(value) -> bool:
    return value is not None and str(value).lower() in ("true", "1")
