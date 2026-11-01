import csv
import re
from pathlib import Path

import pytest
from rdflib import Graph

import kroma
from kroma.cli import main
from kroma.matcher import LexicalEmbedder, build_prompt, find_candidates, normalize
from kroma.ontology import local_name

DATA = Path(__file__).parent / "data"
A = DATA / "anatomy_a.ttl"
B = DATA / "anatomy_b.owl"
TRUE_PAIRS = {
    ("organ", "body organ"),
    ("heart valve", "cardiac valve"),
    ("kidney", "renal organ"),
    ("lung", "lungs"),
}


class FakeJudge:
    """Says Yes, with confidence 9, exactly for the pairs in TRUE_PAIRS."""

    def __init__(self):
        self.calls = []

    def __call__(self, system, user):
        src = re.search(r"Source Concept Names: (.*)", user).group(1).split(", ")[0]
        tgt = re.search(r"Target Concept Names: (.*)", user).group(1).split(", ")[0]
        self.calls.append((src, tgt))
        yes = (src, tgt) in TRUE_PAIRS
        return f"<answer>{'Yes' if yes else 'No'}</answer><confidence>{9 if yes else 7}</confidence>"


def test_load_ontology_reads_labels_synonyms_and_hierarchy():
    a = kroma.load_ontology(A)
    assert len(a) == 5  # the deprecated class is dropped
    heart = a["http://example.org/a#Heart"]
    assert heart.parents == ["http://example.org/a#Organ"]
    assert heart.children == ["http://example.org/a#HeartValve"]
    assert a["http://example.org/a#Kidney"].labels == ["kidney", "Niere"]  # English first
    assert a["http://example.org/a#Lung"].synonyms == ["pulmo"]


def test_load_owl_xml_and_derives_missing_labels():
    b = kroma.load_ontology(B)
    assert len(b) == 5
    assert b["http://example.org/b#CardiacValve"].label == "cardiac valve"
    assert b["http://example.org/b#Cardiac_Organ"].synonyms == ["Heart"]
    assert kroma.load_ontology(Graph().parse(B, format="xml")).concepts.keys() == b.concepts.keys()


def test_local_name_and_normalize():
    assert local_name("http://x.org/o#HeartValve") == "heart valve"
    assert local_name("http://x.org/o/blood_vessel/") == "blood vessel"
    assert normalize(" Heart_Valve (organ)") == "heart valve organ"


def test_find_candidates_ranks_the_true_target_first():
    a, b = kroma.load_ontology(A), kroma.load_ontology(B)
    cands = find_candidates(a, b, top_k=2)
    assert cands["http://example.org/a#Lung"][0][0] == "http://example.org/b#Pulmonary_Organ"
    # "cardiac organ" has the synonym "Heart", so it competes with "cardiac valve"
    assert "http://example.org/b#CardiacValve" in [u for u, _ in cands["http://example.org/a#HeartValve"]]
    assert all(len(v) == 2 for v in cands.values())
    sims = [s for _, s in cands["http://example.org/a#Lung"]]
    assert sims == sorted(sims, reverse=True)


def test_find_candidates_accepts_a_custom_embedder():
    class Constant:
        def encode(self, texts):
            return [[1.0, 0.0] for _ in texts]

    cands = find_candidates(kroma.load_ontology(A), kroma.load_ontology(B), top_k=5, embedder=Constant())
    assert all(len(v) == 5 and v[0][1] == pytest.approx(1.0) for v in cands.values())


def test_lexical_embedder_rows_are_unit_length():
    vecs = LexicalEmbedder().fit(["heart", "lung"]).encode(["heart", "", "lung"])
    norms = (vecs.multiply(vecs).sum(axis=1)).A.ravel()
    assert norms == pytest.approx([1.0, 0.0, 1.0])


def test_prompt_shows_names_and_context():
    a, b = kroma.load_ontology(A), kroma.load_ontology(B)
    _, user = build_prompt(a["http://example.org/a#Heart"], b["http://example.org/b#Cardiac_Organ"], a, b)
    assert "Source Concept Names: heart" in user
    assert "Parent Concepts of the Source Concept: organ" in user
    assert "Child Concepts of the Source Concept: heart valve" in user
    assert "Synonym Concepts of the Target Concept: Heart" in user


def test_match_with_exact_names_and_a_judge():
    judge = FakeJudge()
    result = kroma.match(A, B, llm=judge, workers=2)
    found = {(x.source_label, x.target_label, x.method) for x in result}
    assert found == {
        ("organ", "body organ", "llm"),
        ("heart", "cardiac organ", "exact"),  # matched through the synonym "Heart"
        ("heart valve", "cardiac valve", "llm"),
        ("lung", "lungs", "llm"),
    }  # "kidney"/"renal organ" share no characters: lexical retrieval cannot find it
    assert result.exact_matches == 1
    assert result.llm_calls == len(judge.calls) == result.pairs_judged
    assert ("heart", "cardiac organ") not in judge.calls
    # results come back in source order
    assert [x.source_label for x in result] == ["heart", "heart valve", "lung", "organ"]
    # kidney has no candidate with any similarity, so the model is never asked about it
    assert not any(src == "kidney" for src, _ in judge.calls)


def test_a_semantic_embedder_finds_pairs_without_shared_characters():
    class Semantic:
        """Toy embedder: kidney and renal organ point the same way."""
        def encode(self, texts):
            return [[1.0, 0.0] if t in ("kidney", "renal organ") else [0.0, 0.0] for t in texts]

    result = kroma.match(A, B, llm=FakeJudge(), embedder=Semantic(), top_k=1, exact_match=False)
    assert ("kidney", "renal organ") in {(x.source_label, x.target_label) for x in result}


def test_match_options_reduce_or_widen_the_search():
    judge = FakeJudge()
    assert len(kroma.match(A, B, llm=judge, limit=2)) == 2
    assert len(kroma.match(A, B, llm=FakeJudge(), min_confidence=10)) == 1  # only the exact match
    none = kroma.match(A, B, llm=FakeJudge(), min_similarity=1.1, exact_match=False)
    assert len(none) == 0 and none.llm_calls == 0


def test_match_uses_generate_and_counts_tokens():
    class Model:
        def generate(self, system, prompt, include_tokens=False, truncate=True):
            return "<answer>No</answer><confidence>8</confidence>", 100, 5

    result = kroma.match(A, B, llm=Model(), top_k=1, exact_match=False)
    assert len(result) == 0
    assert result.input_tokens == 100 * result.llm_calls
    assert result.output_tokens == 5 * result.llm_calls


def test_save_alignments_csv_tsv_rdf(tmp_path):
    result = kroma.match(A, B, llm=FakeJudge())
    rows = list(csv.DictReader(result.save(tmp_path / "out.csv").open()))
    assert len(rows) == 4 and rows[0]["relation"] == "="
    assert "\t" in (result.save(tmp_path / "out.tsv")).read_text().splitlines()[0]
    rdf = Graph().parse(result.save(tmp_path / "out.rdf"), format="xml")
    assert len(rdf) > 0
    with pytest.raises(ValueError):
        result.save(tmp_path / "out.json")


def test_cli_match(tmp_path, monkeypatch, capsys):
    import kroma.matcher

    monkeypatch.setattr(kroma.matcher._Judge, "_model", lambda self: FakeJudge())
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr("importlib.util.find_spec", lambda name: True)
    out = tmp_path / "a.csv"
    assert main(["match", str(A), str(B), "-o", str(out), "--workers", "1"]) == 0
    assert len(list(csv.DictReader(out.open()))) == 4
    assert "4 alignments (1 exact)" in capsys.readouterr().out


def test_cli_match_explains_a_missing_key(monkeypatch):
    monkeypatch.delenv("TOGETHERAI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="Set TOGETHERAI_API_KEY"):
        main(["match", str(A), str(B), "--backend", "togetherai"])


def test_neural_embedders_are_called_in_batches():
    from kroma.ontology import Concept, Ontology

    big = Ontology({f"u{i}": Concept(f"u{i}", f"concept {i}", [f"concept {i}"]) for i in range(300)})
    sizes = []

    class Recording:
        def encode(self, texts):
            sizes.append(len(texts))
            return [[1.0, float(len(t))] for t in texts]

    cands = find_candidates(big, big, top_k=1, embedder=Recording())
    assert len(cands) == 300
    assert max(sizes) <= 128 and sum(sizes) == 600
