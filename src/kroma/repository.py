"""Writers for prediction JSONL files and expert-review CSV files."""

import json
import os
from pathlib import Path

from kroma.utils.file import get_predicted_pairs_and_file


class PredictionRepository:
    """Append-only store of predicted alignment rows.

    Rows live in ``results/<mode>/<task>/<agent>_<task>.jsonl``. Existing rows
    are loaded on open so a run can skip pairs it already scored.

    Args:
        task_key (str): Task name from the method config.
        method_name (str): Method name from the method config.
        agent_name (str): Model name; ``/`` is replaced by ``-`` in the file
            name.
        **flags: Run flags forwarded to
            :func:`kroma.utils.file.get_predicted_pairs_and_file`.

    Attributes:
        keys (set[tuple[str, str]]): ``(source, target)`` codes already
            predicted.
        rows (list[dict]): Prediction rows, oldest first.
    """

    def __init__(self, task_key, method_name, agent_name, **flags):
        self.keys, self.rows, self._handle = get_predicted_pairs_and_file(
            task_key,
            method_name,
            agent_name,
            **flags,
        )

    def seen(self, key) -> bool:
        """Return whether a ``(source, target)`` pair was already predicted."""
        return key in self.keys

    def append(self, row: dict) -> None:
        """Write one prediction row and flush it to disk.

        Args:
            row (dict): Prediction with at least ``source`` and ``target``.
        """
        self._handle.write(json.dumps(row) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self.keys.add((row["source"], row["target"]))
        self.rows.append(row)

    @property
    def name(self) -> str:
        """str: Path of the prediction file."""
        return self._handle.name


class ReviewRepository:
    """CSV of pairs sent to an expert after online refinement.

    The file is ``reviews/baseline/<subtask>/expert_queries.csv`` with the
    columns ``source,target,expert_edge``.

    Args:
        subtask (str): Dataset subtask name, used as the folder name.
    """

    def __init__(self, subtask: str):
        review_dir = Path("reviews") / "baseline" / subtask
        review_dir.mkdir(parents=True, exist_ok=True)
        self.path = review_dir / "expert_queries.csv"
        if not self.path.exists():
            self.path.write_text("source,target,expert_edge\n")

    def append(self, src_key, tgt_key, expert_qs) -> None:
        """Append the expert questions raised by one pair.

        Args:
            src_key (tuple[str, str]): Source ``(code, uri)``.
            tgt_key (tuple[str, str]): Target ``(code, uri)``.
            expert_qs (list[tuple] | None): Edges returned by
                :func:`kroma.algorithms.refinement.online_refine`.
        """
        with self.path.open("a+") as handle:
            for src, tgt in expert_qs or []:
                handle.write(f"{src_key},{tgt_key},{src}->{tgt}\n")
