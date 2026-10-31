"""Build the matching task: sample pairs, map hierarchies, gather context."""

import random
import itertools
from collections import defaultdict
from typing import List, Tuple, Dict, Any
from kroma.tasks.query_engine import query


def build_parent_child_map(
    G: Dict[Tuple[str, str], Any]
) -> Tuple[Dict[str, set], Dict[str, set]]:
    """Index a ``(code, uri)``-keyed hierarchy by code.

    Args:
        G (dict): ``(code, uri)`` to iterable of child ``(code, uri)`` keys.

    Returns:
        tuple[dict[str, set], dict[str, set]]: ``(parent_map, child_map)``
        from a code to its parent codes and to its child codes.
    """
    parent_map, child_map = defaultdict(set), defaultdict(set)
    for (label, _), children in G.items():
        child_map[label] = {c[0] for c in children}
        for c in children:
            parent_map[c[0]].add(label)
    return parent_map, child_map


def build_ontology_matching_task(
    G1: Dict[Tuple[str, str], Any],
    G2: Dict[Tuple[str, str], Any],
    alignments: List[Tuple[Any, Any, str]],
    sample_sz: int = 50,
    dictionary: Dict[str, Any] = None,
    query_opts: List[str] = None
) -> Tuple[
    Dict[Tuple[str, str], Any],
    Dict[Tuple[str, str], Any],
    List[Tuple[Any, Any, str]],
    Dict[Tuple[str, str], Dict[str, List[str]]],
    Dict[Tuple[str, str], Dict[str, List[str]]],
    Dict[str, set],
    Dict[str, set]
]:
    """Sample alignment pairs and gather context for every concept.

    Up to ``sample_sz`` positives and ``sample_sz`` negatives are sampled and
    shuffled. When the alignments contain no negatives, every unseen
    source/target pair is used as a negative candidate.

    Args:
        G1 (dict): Source graph from a loader.
        G2 (dict): Target graph from a loader.
        alignments (list): ``(source_key, target_key, label)`` triples.
        sample_sz (int): Maximum positives, and maximum negatives, to keep.
        dictionary (dict | None): Cached dictionaries for :func:`query`.
        query_opts (list[str] | None): Context sources for :func:`query`.
            Metadata is only gathered when both this and ``dictionary`` are
            given.

    Returns:
        tuple: ``(G1, G2, sampled, source_meta, target_meta, source_map,
        target_map)``. The metadata dicts map a concept code to
        ``{"parents", "children", "synonyms", "labels"}`` lists; the maps are
        ``{"parent": ..., "child": ...}`` from
        :func:`build_parent_child_map`.
    """
    # sample positives/negatives
    positives = [m for m in alignments if m[2] == "1"]
    negatives = [m for m in alignments if m[2] != "1"]

    if not negatives:
        seen = {(s[0], t[0]) for s, t, _ in alignments}
        negatives = [
            (s, t, "0")
            for s, t in itertools.product(G1, G2)
            if (s[0], t[0]) not in seen
        ]

    pos_sel = random.sample(positives, min(len(positives), sample_sz))
    neg_sel = random.sample(negatives, min(len(negatives), sample_sz))
    sampled = pos_sel + neg_sel
    random.shuffle(sampled)

    # build parent/child maps
    pm1, cm1 = build_parent_child_map(G1)
    pm2, cm2 = build_parent_child_map(G2)

    # enrich metadata using URI lookup
    OS_meta: Dict[Tuple[str, str], Dict[str, List[str]]] = {}
    OT_meta: Dict[Tuple[str, str], Dict[str, List[str]]] = {}
    if dictionary and query_opts:
        for key in G1:
            lookup_key = key[0]  # use source URI as the dictionary lookup
            p, c, s, l = query(
                lookup_key,
                {'parent': pm1, 'child': pm1},
                dictionary,
                query_opts
            )
            OS_meta[lookup_key] = {'parents': p, 'children': c, 'synonyms': s, 'labels': l}

        for key in G2:
            lookup_key = key[0]
            p, c, s, l = query(
                lookup_key,
                {'parent': pm2, 'child': pm2},
                dictionary,
                query_opts
            )
            OT_meta[lookup_key] = {'parents': p, 'children': c, 'synonyms': s, 'labels': l}

    return (
        G1,
        G2,
        sampled,
        OS_meta,
        OT_meta,
        {'parent': pm1, 'child': cm1},
        {'parent': pm2, 'child': cm2}
    )
