"""Match two ontologies: :func:`match` takes two ontologies and returns alignments.

For each source class KROMA

1. **retrieves** the most similar target classes by comparing all their labels
   and synonyms in embedding space,
2. **accepts** a candidate outright when a label or synonym is identical
   (after normalising case, spacing and punctuation), and otherwise
3. **asks a language model** whether the two classes are the same, showing it
   both classes' labels, synonyms, parents and children, and keeps the
   answers that say Yes.

This is the retrieval-and-judging core of KROMA as a library call. The
benchmark pipeline in :mod:`kroma.pipeline` adds dataset configs, debate,
active learning and graph refinement on top.
"""

import logging
import re
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import sparse

from kroma.agents.utils import extract_confidence, extract_yes_no
from kroma.alignment import Alignment, MatchResult
from kroma.ontology import Concept, Ontology, load_ontology
from kroma.tasks.prompts.builders import list_to_str
from kroma.tasks.prompts.templates import SYSTEM_PROMPT, TASK_PROMPT

logger = logging.getLogger(__name__)

DEFAULT_BACKEND = "openai"
DEFAULT_MODEL = "gpt-4o-mini"
CONTEXT_LIMIT = 10


class LexicalEmbedder:
    """Character n-gram TF-IDF vectors; a fast embedder that needs no model.

    Strings that share many character n-grams get similar vectors, which is a
    strong signal for ontology labels. Call :meth:`fit` on the corpus first so
    rare n-grams weigh more; unfitted, all n-grams weigh the same.

    Args:
        n (int): N-gram length.
        dim (int): Hashing dimension.

    Example:
        >>> vecs = LexicalEmbedder().encode(["heart valve", "valve of heart", "kidney"])
        >>> sims = (vecs @ vecs.T).toarray()
        >>> bool(sims[0, 1] > sims[0, 2])
        True
    """

    def __init__(self, n: int = 3, dim: int = 2 ** 20):
        self.n = n
        self.dim = dim
        self.idf: Optional[np.ndarray] = None

    def _grams(self, text: str) -> List[int]:
        text = normalize(text)
        if not text:
            return []
        text = f" {text} "
        return [zlib.crc32(text[i:i + self.n].encode()) % self.dim
                for i in range(max(1, len(text) - self.n + 1))]

    def fit(self, texts: Sequence[str]) -> "LexicalEmbedder":
        """Compute inverse document frequencies from ``texts``."""
        df = np.zeros(self.dim, dtype=np.float32)
        for text in texts:
            df[list(set(self._grams(text)))] += 1
        self.idf = np.log((1 + len(texts)) / (1 + df)).astype(np.float32) + 1
        return self

    def encode(self, texts: Sequence[str]) -> sparse.csr_matrix:
        """Return one L2-normalised sparse row per string."""
        rows, cols = [], []
        for row, text in enumerate(texts):
            grams = self._grams(text)
            rows.extend([row] * len(grams))
            cols.extend(grams)
        data = np.ones(len(cols), dtype=np.float32)
        out = sparse.csr_matrix((data, (rows, cols)), shape=(len(texts), self.dim))
        out.sum_duplicates()
        if self.idf is not None:
            out.data *= self.idf[out.indices]
        norms = np.sqrt(np.asarray(out.multiply(out).sum(axis=1)).ravel())
        return sparse.diags(1 / np.where(norms == 0, 1, norms)).dot(out).tocsr()


def normalize(text: str) -> str:
    """Lower-case, turn punctuation and underscores into spaces, and collapse whitespace.

    Example:
        >>> normalize("  Heart_Valve (organ) ")
        'heart valve organ'
    """
    return " ".join(re.sub(r"[\W_]+", " ", text.lower()).split())


def find_candidates(
    source: Ontology,
    target: Ontology,
    top_k: int = 5,
    embedder: Any = "lexical",
    batch_size: int = 256,
) -> Dict[str, List[Tuple[str, float]]]:
    """Retrieve the ``top_k`` most similar target classes for every source class.

    The similarity of two classes is the highest cosine similarity between
    any of their names (labels and synonyms).

    Args:
        source (Ontology): Source ontology.
        target (Ontology): Target ontology.
        top_k (int): Candidates per source class.
        embedder: ``"lexical"`` for :class:`LexicalEmbedder`;
            ``"huggingface:<model>"`` or ``"sentence-transformers:<model>"``
            for a neural model (needs the ``models`` extra); or any object
            with an ``encode(list[str])`` method returning vectors.
        batch_size (int): Source classes compared per batch.

    Returns:
        dict[str, list[tuple[str, float]]]: Source IRI to ``(target IRI,
        similarity)`` pairs, most similar first.
    """
    src = list(source)
    tgt = list(target)
    if not src or not tgt:
        return {c.uri: [] for c in src}
    src_names, src_owner = _flatten(src)
    tgt_names, tgt_owner = _flatten(tgt)
    model = _resolve_embedder(embedder, src_names + tgt_names)
    src_vecs = _unit(_encode(model, src_names))
    tgt_vecs = _unit(_encode(model, tgt_names))
    tgt_t = tgt_vecs.T.tocsr() if sparse.issparse(tgt_vecs) else tgt_vecs.T
    tgt_starts = np.flatnonzero(np.r_[True, tgt_owner[1:] != tgt_owner[:-1]])
    src_starts = np.flatnonzero(np.r_[True, src_owner[1:] != src_owner[:-1]])
    src_ends = np.r_[src_starts[1:], len(src_owner)]
    k = min(top_k, len(tgt))

    result: Dict[str, List[Tuple[str, float]]] = {}
    for first in range(0, len(src), batch_size):
        last = min(first + batch_size, len(src))
        rows = slice(src_starts[first], src_ends[last - 1])
        sims = src_vecs[rows] @ tgt_t                          # names x target names
        sims = sims.toarray() if sparse.issparse(sims) else np.asarray(sims)
        per_target = np.maximum.reduceat(sims, tgt_starts, axis=1)  # names x targets
        offsets = src_starts[first:last] - src_starts[first]
        per_pair = np.maximum.reduceat(per_target, offsets, axis=0)  # sources x targets
        top = np.argpartition(-per_pair, k - 1, axis=1)[:, :k]
        for i, idx in enumerate(top):
            order = idx[np.argsort(-per_pair[i, idx], kind="stable")]
            result[src[first + i].uri] = [(tgt[j].uri, float(per_pair[i, j])) for j in order]
    return result


def build_prompt(src: Concept, tgt: Concept, source: Ontology, target: Ontology) -> Tuple[str, str]:
    """Build the system and user prompts that ask whether two classes match.

    Args:
        src (Concept): Source class.
        tgt (Concept): Target class.
        source (Ontology): Ontology of ``src``, to name its parents and children.
        target (Ontology): Ontology of ``tgt``.

    Returns:
        tuple[str, str]: ``(system_prompt, user_prompt)``.
    """
    def ctx(uris, onto):
        return list_to_str(onto.labels_of(uris)[:CONTEXT_LIMIT])

    user = TASK_PROMPT.format(
        demonstrations="",
        src_names=list_to_str(src.labels),
        src_parents=ctx(src.parents, source),
        src_children=ctx(src.children, source),
        src_synonyms=list_to_str(src.synonyms[:CONTEXT_LIMIT]),
        tgt_names=list_to_str(tgt.labels),
        tgt_parents=ctx(tgt.parents, target),
        tgt_children=ctx(tgt.children, target),
        tgt_synonyms=list_to_str(tgt.synonyms[:CONTEXT_LIMIT]),
    )
    return SYSTEM_PROMPT, user


def match(
    source: Union[str, Ontology, Any],
    target: Union[str, Ontology, Any],
    *,
    llm: Any = None,
    backend: str = DEFAULT_BACKEND,
    model: str = DEFAULT_MODEL,
    top_k: int = 3,
    min_similarity: float = 0.01,
    min_confidence: int = 0,
    exact_match: bool = True,
    first_match_only: bool = True,
    embedder: Any = "lexical",
    workers: int = 4,
    limit: Optional[int] = None,
) -> MatchResult:
    """Align two ontologies.

    Args:
        source: Source ontology: a path or URL of an RDF file (``.owl``,
            ``.ttl``, ...), an :class:`rdflib.Graph`, or an
            :class:`~kroma.ontology.Ontology`.
        target: Target ontology, in the same forms.
        llm: The judge. ``None`` creates a chat model from ``backend`` and
            ``model`` (reading ``<BACKEND>_API_KEY``). Otherwise either an
            object with KROMA's ``generate(system_prompt, prompt, ...)`` method
            (see :class:`~kroma.model_providers.base_model.BaseModel`) or a
            plain callable ``llm(system_prompt, user_prompt) -> str``.
        backend (str): Chat backend when ``llm`` is ``None``: ``"openai"``,
            ``"togetherai"`` or a registered one.
        model (str): Model name when ``llm`` is ``None``.
        top_k (int): Candidate targets considered per source class.
        min_similarity (float): Skip candidates below this retrieval
            similarity, saving model calls. The default skips only pairs with
            nothing in common. Names that share no characters (such as
            "kidney" and "renal organ") need a neural ``embedder`` to be found.
        min_confidence (int): Keep a Yes only at this confidence (0-10) or
            higher.
        exact_match (bool): Accept identical names without asking the model.
        first_match_only (bool): Stop at the first accepted candidate of a
            source class, giving at most one target per source.
        embedder: Retrieval embedder; see :func:`find_candidates`.
        workers (int): Source classes judged in parallel.
        limit (int | None): Only match the first ``limit`` source classes,
            useful for a trial run.

    Returns:
        MatchResult: Accepted alignments and call counts.

    Example:
        >>> import kroma
        >>> result = kroma.match("mouse.owl", "human.owl",
        ...                      backend="openai", model="gpt-4o-mini")  # doctest: +SKIP
        >>> result.save("mouse-human.csv")  # doctest: +SKIP
    """
    src_onto = load_ontology(source)
    tgt_onto = load_ontology(target)
    logger.info("Matching %r against %r", src_onto, tgt_onto)
    candidates = find_candidates(src_onto, tgt_onto, top_k=top_k, embedder=embedder)
    sources = list(src_onto)[:limit] if limit is not None else list(src_onto)
    tgt_index = _name_index(tgt_onto) if exact_match else {}
    judge = _Judge(llm, backend, model)

    def work(src: Concept) -> MatchResult:
        out = MatchResult()
        cands = candidates.get(src.uri, [])
        exact = _exact_targets(src, tgt_index)
        cands = cands + [(uri, 1.0) for uri in sorted(exact - {u for u, _ in cands})]
        for tgt_uri, sim in sorted(cands, key=lambda c: c[0] not in exact):
            tgt = tgt_onto[tgt_uri]
            if tgt_uri in exact:
                out.alignments.append(Alignment(src.uri, tgt.uri, src.label, tgt.label, 10, sim, "exact"))
                out.exact_matches += 1
            elif sim < min_similarity:
                continue
            else:
                answer, confidence = judge(out, *build_prompt(src, tgt, src_onto, tgt_onto))
                out.pairs_judged += 1
                if not (answer == 1 and confidence >= min_confidence):
                    continue
                out.alignments.append(Alignment(src.uri, tgt.uri, src.label, tgt.label, confidence, sim, "llm"))
            if first_match_only:
                break
        return out

    result = MatchResult()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for done, part in enumerate(pool.map(work, sources), 1):
            result.alignments.extend(part.alignments)
            for name in ("pairs_judged", "exact_matches", "llm_calls", "input_tokens", "output_tokens"):
                setattr(result, name, getattr(result, name) + getattr(part, name))
            if done % 100 == 0 or done == len(sources):
                logger.info("%d/%d source classes, %d alignments", done, len(sources), len(result))
    return result


class _Judge:
    """Wraps the configured model and parses its answer; created lazily."""

    def __init__(self, llm, backend, model):
        self.llm, self.backend, self.model = llm, backend, model
        self._lock = threading.Lock()

    def _model(self):
        with self._lock:
            if self.llm is None:
                from kroma.inference.factory import create_inference_model
                self.llm = create_inference_model(self.backend, self.model)
        return self.llm

    def __call__(self, stats: MatchResult, system: str, user: str) -> Tuple[int, int]:
        llm = self._model()
        if hasattr(llm, "generate"):
            reply, tokens_in, tokens_out = llm.generate(system, user, include_tokens=True, truncate=False)
            stats.input_tokens += tokens_in or 0
            stats.output_tokens += tokens_out or 0
        else:
            reply = llm(system, user)
        stats.llm_calls += 1
        reply = str(reply)
        return extract_yes_no(reply), extract_confidence(reply)


def _flatten(concepts: List[Concept]) -> Tuple[List[str], np.ndarray]:
    names, owner = [], []
    for i, concept in enumerate(concepts):
        for name in concept.names or [concept.label]:
            names.append(name)
            owner.append(i)
    return names, np.asarray(owner)


def _encode(model, texts: List[str], batch_size: int = 128):
    """Encode in batches, so a neural model never sees the whole ontology at once."""
    if isinstance(model, LexicalEmbedder):
        return model.encode(texts)
    parts = [np.asarray(model.encode(texts[i:i + batch_size]), dtype=np.float32)
             for i in range(0, len(texts), batch_size)]
    return np.vstack(parts)


def _unit(vectors):
    if sparse.issparse(vectors):
        return vectors.tocsr()
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def _resolve_embedder(embedder, corpus):
    if embedder in (None, "lexical"):
        return LexicalEmbedder().fit(corpus)
    if isinstance(embedder, str):
        backend, _, name = embedder.partition(":")
        from kroma.inference.factory import create_embedding_model
        return create_embedding_model(backend, name) if name else create_embedding_model(backend)
    return embedder


def _name_index(onto: Ontology) -> Dict[str, set]:
    index: Dict[str, set] = {}
    for concept in onto:
        for name in concept.names:
            index.setdefault(normalize(name), set()).add(concept.uri)
    return index


def _exact_targets(src: Concept, index: Dict[str, set]) -> set:
    found = set()
    for name in src.names:
        found |= index.get(normalize(name), set())
    return found
