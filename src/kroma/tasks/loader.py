"""Loaders that turn benchmark files into graphs and candidate alignments.

A loader returns ``(source_graph, target_graph, alignments)``. Graph keys are
``(code, uri)`` tuples mapped to their children; each alignment is
``((src_code, src_uri), (tgt_code, tgt_uri), label)`` with ``label`` the
string ``"1"`` for a match and ``"0"`` otherwise.
"""

import csv
import ast
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Any


class OntologyLoader(ABC):
    """Abstract base for ontology data loaders."""
    @abstractmethod
    def load(self) -> Tuple[
        Dict[Tuple[str, str], Any],  # source graph
        Dict[Tuple[str, str], Any],  # target graph
        List[Tuple[Tuple[str, str], Tuple[str, str], str]]  # alignments
    ]:
        """Load the data.

        Returns:
            tuple: ``(source_graph, target_graph, alignments)`` as described
            in :mod:`kroma.tasks.loader`.
        """
        pass


class CsvAlignmentLoader(OntologyLoader):
    """Loads pairwise alignments from CSV benchmark files.

    Two layouts are supported:

    * ``ncit-doid``: columns ``src_code``, ``src_ety``, ``tgt_code``,
      ``tgt_ety``, ``score`` and an optional ``tgt_cands`` list. Each
      candidate becomes one alignment, labelled ``"1"`` only for the true
      target when ``score`` is ``1``.
    * any other value: columns ``source``, ``source_uri``, ``target``,
      ``target_uri`` and ``relation`` (or ``Relation``, default ``"1"``).

    Args:
        csv_paths (list[str | pathlib.Path]): CSV files to read, in order.
        dataset (str): Layout name, ``"ncit-doid"`` or another dataset type.
    """
    def __init__(self, csv_paths: List[str], dataset: str = "ncit-doid"):
        self.csv_paths = csv_paths
        self.dataset = dataset

    def load(self) -> Tuple[
        Dict[Tuple[str, str], Any],
        Dict[Tuple[str, str], Any],
        List[Tuple[Tuple[str, str], Tuple[str, str], str]]
    ]:
        """Read every CSV file.

        Returns:
            tuple: ``(source_graph, target_graph, alignments)``. The graphs
            contain every concept that appears in a row, with empty child
            lists.
        """
        G1 = defaultdict(list)
        G2 = defaultdict(list)
        alignments = []

        for path in self.csv_paths:
            reader = csv.DictReader(Path(path).open(encoding="utf-8"))
            for row in reader:
                if self.dataset == 'ncit-doid':
                    s_lbl, s_uri = row["src_code"].strip(), row["src_ety"].strip()
                    t_lbl, t_uri = row["tgt_code"].strip(), row["tgt_ety"].strip()
                    candidates = ast.literal_eval(row.get("tgt_cands", "[]"))
                    truth = row.get("score", "0").strip()
                    for cand in candidates or [t_uri]:
                        lbl = '1' if (cand == t_uri and truth == '1') else '0'
                        cand_lbl = cand.split('#')[-1] if '#' in cand else cand
                        alignments.append(((s_lbl, s_uri), (cand_lbl, cand), lbl))
                else:
                    s_lbl, s_uri = row.get('source',''), row.get('source_uri','')
                    t_lbl, t_uri = row.get('target',''), row.get('target_uri','')
                    lbl = row.get('Relation', row.get('relation','1')).strip()
                    alignments.append(((s_lbl, s_uri), (t_lbl, t_uri), lbl))

                # ensure nodes appear
                G1.setdefault((s_lbl, s_uri), [])
                G2.setdefault((t_lbl, t_uri), [])

        return G1, G2, alignments
