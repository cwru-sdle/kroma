"""Load configs and caches, and open prediction files."""

import json
import os
from pathlib import Path
from typing import Tuple, Set, TextIO, Dict, Any, List

def load_config(filepath: str) -> Any:
    """Load a JSON config file.

    The file is parsed as a single JSON document, so a ``.jsonl`` method
    config must hold one object.

    Args:
        filepath (str | pathlib.Path): Config path.

    Returns:
        Any: The parsed JSON.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    with open(path, "r") as file:
        data = json.load(file)
    return data


def load_cache(filepath: str, key_field: str = 'concept_code') -> Dict[str, Dict[str, List[Any]]]:
    """Load a JSONL context cache, merging lines that share a key.

    Each line is an object with ``key_field`` and optional ``parents``,
    ``children``, ``synonyms`` and ``labels`` lists. Lines without the key
    are skipped.

    Args:
        filepath (str | pathlib.Path): Cache path.
        key_field (str): Field holding the concept code.

    Returns:
        dict[str, dict[str, list]]: Concept code to deduplicated lists (in no
        particular order).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    cache: Dict[str, Dict[str, Set[Any]]] = {}
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {filepath}")

    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = record.get(key_field)
            if key is None:
                continue
            if key not in cache:
                cache[key] = {field: set() for field in ['parents','children','synonyms','labels']}
            for field in ['parents','children','synonyms','labels']:
                values = record.get(field)
                if isinstance(values, list):
                    cache[key][field].update(values)
    # convert sets back to lists
    return {k: {fld: list(vset) for fld, vset in fields.items()} for k, fields in cache.items()}


def get_predicted_pairs_and_file(
    task_name: str,
    method_name: str,
    agent_name: str = None,
    debate: bool = False,
    size: str = None,
    reasoning: bool = False,
    baseline: bool = False,
    active_learning: bool = False,
    compare_models: bool = False,
    bisim: bool = True
) -> Tuple[Set[Tuple[str,str]], TextIO]:
    """Open the prediction file for a run and load what it already holds.

    The file is ``results/<mode>/<task_name>/<agent>_<task_name>.jsonl``
    where ``mode`` is ``baseline`` or ``results``. The remaining flags are
    accepted for callers and do not change the path.

    Args:
        task_name (str): Task name.
        method_name (str): Method name.
        agent_name (str): Model name; ``/`` becomes ``-``.
        baseline (bool): Write under ``results/baseline``.

    Returns:
        tuple: ``(keys, rows, handle)``: the set of predicted
        ``(source, target)`` codes, the existing rows, and the file opened for
        appending.
    """
    # decide mode folder
    mode = "baseline" if baseline else "results"

    # sanitize agent_name to a filesystem‐safe string
    safe_agent = agent_name.replace("/", "-")

    # build directory: results/<mode>/<task_name>/
    results_dir = Path("results") / mode / task_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # build filename and full path
    filename = f"{safe_agent}_{task_name}.jsonl"
    filepath = results_dir / filename

    # ensure the file exists
    filepath.touch(exist_ok=True)

    # load existing predictions
    keys: Set[Tuple[str,str]] = set()
    predicted: List[Dict] = []
    for line in filepath.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            key = (rec['source'], rec['target'])
            if key not in keys:
                keys.add(key)
                predicted.append(rec)
        except json.JSONDecodeError:
            continue

    # open for append+read and seek to end
    handle = filepath.open("a+", encoding="utf-8")
    handle.seek(0, os.SEEK_END)

    print(f"Using prediction file: {filepath}")
    print(keys)
    print(predicted)
    return keys, predicted, handle