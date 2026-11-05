"""`kroma run`'s pipeline, end to end on a small track, with a scripted model and embedder."""

import csv
import json
import re
from types import SimpleNamespace as Reply

import numpy as np
import pytest

pytest.importorskip("node2vec")

from kroma.cli import build_run_parser  # noqa: E402
from kroma.inference.factory import CHAT_BACKENDS  # noqa: E402
from kroma.model_providers.providers import ChatProvider, parse_chat_response  # noqa: E402
from kroma.pipeline import MatchingRun  # noqa: E402
from kroma.tasks.query_engine import QUERY_STRATEGIES  # noqa: E402

PAIRS = [("heart", "cardiac organ"), ("lung", "lungs"), ("kidney", "renal organ"), ("liver", "hepatic organ"),
         ("organ", "body organ"), ("spleen", "splenic organ"), ("brain", "cerebral organ"), ("vein", "veins")]
FAMILY = {"organ": ["heart", "lung", "kidney", "liver", "spleen", "brain"], "body organ": ["cardiac organ", "lungs"]}


class Scripted(ChatProvider):
    def __init__(self, api_key):
        pass

    def create_completion(self, model, messages, temperature, max_tokens, response_format=None):
        text = messages[-1]["content"]
        found = re.search(r"source concept “([^”]*)” has.*?target concept “([^”]*)” has", text, re.S)
        answer = "yes" if (found[1], found[2]) in PAIRS else "no"
        content = f"<answer> {answer} </answer>\n<confidence>9</confidence>"
        return Reply(choices=[Reply(message=Reply(content=content))], usage=Reply(prompt_tokens=1, completion_tokens=1))

    def parse_response(self, response, include_tokens):
        return parse_chat_response(response, include_tokens)


class Family:
    def fetch(self, code, pcmap, dictionary, topk=3):
        parents = [p for p, kids in FAMILY.items() if code in kids]
        return parents, FAMILY.get(code, []), [], [code]


class Embedder:
    def encode(self, texts):
        vectors = np.array([[t.count(chr(97 + i)) + 1.0 for i in range(26)] for t in texts])
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def write_track(root, sample_sz):
    configs = root / "experiments" / "configs"
    (configs / "method" / "scripted").mkdir(parents=True)
    (configs / "datasets").mkdir()
    (root / "data").mkdir()
    with (root / "data" / "track.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "source_uri", "target", "target_uri", "Relation"])
        for s, t in PAIRS:
            writer.writerow([s, f"http://a/{s}", t, f"http://b/{t}", "1"])
    (root / "data" / "empty.jsonl").write_text("{}\n")
    (configs / "dictionary.json").write_text(json.dumps({"none": "data/empty.jsonl"}))
    (configs / "datasets" / "track.json").write_text(json.dumps({
        "subtask": "track", "task_name": "ontology_matching", "datasets": ["track"], "dataset_type": "csv",
        "csv_folder": "./data", "sample_sz": sample_sz,
    }))
    (configs / "method" / "scripted" / "track.jsonl").write_text(json.dumps({
        "agent_type": "scripted", "agent_name": "scripted", "task": "track", "method_name": "kroma",
        "query_options": ["family"],
    }))


@pytest.fixture
def track(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SCRIPTED_API_KEY", "x")
    monkeypatch.setitem(CHAT_BACKENDS, "scripted", Scripted)
    monkeypatch.setitem(QUERY_STRATEGIES, "family", Family())
    monkeypatch.setattr("kroma.pipeline.create_embedding_model", lambda *_: Embedder())
    monkeypatch.setattr("kroma.pipeline.NODE2VEC_DEFAULTS", {"dimensions": 8, "walk_length": 5, "num_walks": 5, "workers": 1})
    return tmp_path


def run(*flags):
    return MatchingRun(build_run_parser().parse_args(["--method_config", "track", "--llm", "scripted", *flags]))


def test_a_run_finishes_and_scores_the_model(track):
    write_track(track, sample_sz=4)
    pipeline = run()
    summary = pipeline.execute()
    assert len(pipeline.y_true) == 8
    assert summary["metrics"]["f1"] == 1.0


def test_the_hierarchy_comes_from_concept_metadata(track):
    write_track(track, sample_sz=4)
    pipeline = run()
    pipeline.prepare()
    assert ("heart", "http://a/heart") in pipeline.source[("organ", "http://a/organ")]
    assert ("lungs", "http://b/lungs") in pipeline.target[("body organ", "http://b/body organ")]


def test_retrieval_context_leaves_out_the_concepts_being_scored(track):
    write_track(track, sample_sz=1)
    pipeline = run()
    pipeline.prepare()
    scored = {s[0] for s, _, _ in pipeline.task_aligns}
    query = next(iter(pipeline.source_vectors.values()))
    stored = {key[0] for key, _ in pipeline.src_store.query(query, top_k=20)}
    assert stored and not stored & scored


def test_a_track_with_every_concept_scored_still_runs(track):
    write_track(track, sample_sz=100)
    assert run().execute()["metrics"]["f1"] >= 0.0


def test_a_chosen_refiner_is_used(track, monkeypatch):
    from kroma.algorithms.refiners import BisimulationRefiner, REFINERS

    calls = []

    class Spy(BisimulationRefiner):
        def offline(self, adj, rank_attr):
            calls.append("offline")
            return super().offline(adj, rank_attr)

    monkeypatch.setitem(REFINERS, "spy", Spy)
    write_track(track, sample_sz=4)
    run("--refiner", "spy").execute()
    assert calls


def test_a_graph_with_no_edges_gets_zero_structure():
    from kroma.algorithms.node2vec import compute_combined_embeddings

    text = np.ones((2, 3))
    combined = compute_combined_embeddings({"a": [], "b": []}, ["a", "b"], text, n2v_kwargs={"dimensions": 4})
    assert combined.shape == (2, 7)
