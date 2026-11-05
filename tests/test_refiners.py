"""Pluggable refinement, and relation refinement on triples."""

import json

import pytest

from kroma.algorithms.refinement import offline_refine
from kroma.algorithms.refiners import BisimulationRefiner, REFINERS, create_refiner, register_refiner
from kroma.algorithms.relation_rules import canonicalize_predicate, load_rules
from kroma.algorithms.relations import RelationRefiner, RefinementOptions


def test_the_default_is_the_paper_method():
    adj = {"a": ["b", "c"], "b": [], "c": []}
    ranks = {"a": 1, "b": 0, "c": 0}
    assert create_refiner().offline(adj, ranks) == offline_refine(adj, ranks)


def test_online_refinement_goes_through_the_refiner():
    graph = {"a": {"b"}, "b": set()}
    classes, questions = create_refiner().online(graph, {"a": 1, "b": 0}, {"a": "a", "b": "b"}, [("a", "b")], 1)
    assert classes["a"] == classes["b"]
    assert questions == []


def test_a_refiner_can_be_registered(monkeypatch):
    class KeepAll(BisimulationRefiner):
        def offline(self, adj, rank_attr):
            return adj

    monkeypatch.setitem(REFINERS, "keep_all", KeepAll)
    adj = {"a": ["b"]}
    assert create_refiner("keep_all").offline(adj, {"a": 1, "b": 0}) is adj
    register_refiner("other", KeepAll)
    assert isinstance(create_refiner("other"), KeepAll)
    del REFINERS["other"]


def test_an_unknown_refiner_says_what_is_available():
    with pytest.raises(ValueError, match="bisimulation"):
        create_refiner("nope")


def test_predicates_reduce_to_their_local_name():
    assert canonicalize_predicate("<http://www.w3.org/2000/01/rdf-schema#subClassOf>") == "subclassof"
    assert canonicalize_predicate("rdfs:subClassOf") == "subclassof"
    assert canonicalize_predicate(None) == ""


def test_rule_loading_merges_all_fields_and_normalizes(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text(
        json.dumps(
            {
                "normalize": {"contains": "has_part"},
                "inverse": {"has_part": "part_of"},
                "transitive": ["contains"],
                "drop": ["ignore_me"],
                "grammar": {"mentors": "is_mentored_by"},
                "grammar_flip": ["is_mentored_by"],
                "domain_range": {
                    "teaches": {"domain": ["Person"], "range": ["Course"]}
                },
            }
        )
    )

    rules = load_rules(path)

    assert rules.normalize_triple(("whole", "contains", "piece")) == (
        "piece",
        "part_of",
        "whole",
    )
    assert rules.normalize_triple(("teacher", "mentors", "student")) == (
        "student",
        "is_mentored_by",
        "teacher",
    )
    assert rules.normalize_triple(("a", "ignore_me", "b")) is None
    assert rules.domain_range["teaches"] == ({"Person"}, {"Course"})
    assert "subclassof" in rules.transitive
    assert "part_of" in rules.transitive


def test_pipeline_deduplicates_and_computes_transitive_reduction():
    result = RelationRefiner().refine(
        [
            ("A", "is_a", "B"),
            ("A", "is_a", "B"),
            ("B", "is_a", "C"),
            ("A", "is_a", "C"),
        ],
        RefinementOptions(apply_case3=False),
    )

    assert result.triples == [
        ("A", "subclassof", "B"),
        ("B", "subclassof", "C"),
    ]
    assert result.diagnostics[0].code == "triples_removed"
    assert result.metadata["stages"] == [
        "normalize",
        "transitive_closure",
        "transitive_reduction",
    ]
    assert result.actions


def test_pipeline_reports_cycles_and_domain_conflicts_structurally():
    refiner = RelationRefiner()
    cycle = refiner.refine(
        [("A", "is_a", "B"), ("B", "is_a", "A")],
        RefinementOptions(apply_case3=False),
    )
    assert {item.code for item in cycle.diagnostics} == {"cycle_detected"}
    assert len(cycle.triples) == 2

    rules = load_rules()
    rules.domain_range["teaches"] = ({"Person"}, {"Course"})
    mismatch = RelationRefiner(rules).refine(
        [("rock", "teaches", "math")],
        RefinementOptions(
            transitive_closure=False,
            node_domain={"rock": {"Mineral"}, "math": {"Course"}},
        ),
    )
    assert mismatch.triples == []
    assert mismatch.conflicts[0].code == "domain_mismatch"
    assert mismatch.conflicts[0].triple == ("rock", "teaches", "math")


def test_pipeline_bounds_closure_and_reports_factual_denials():
    bounded = RelationRefiner().refine(
        [
            ("A", "is_a", "B"),
            ("B", "is_a", "C"),
            ("C", "is_a", "D"),
        ],
        RefinementOptions(
            apply_case3=False,
            transitive_reduction=False,
            max_new_edges=1,
        ),
    )
    assert len(bounded.triples) == 4
    assert any(item.code == "closure_truncated" for item in bounded.diagnostics)

    denied = RelationRefiner().refine(
        [("A", "unverified_relation", "B")],
        RefinementOptions(
            transitive_closure=False,
            factual_check=lambda *_: False,
        ),
    )
    assert denied.triples == []
    assert {item.code for item in denied.conflicts} == {"factual_denied"}
    assert any(item.code == "non_reversible" for item in denied.diagnostics)

    unchecked = RelationRefiner().refine(
        [("A", "unverified_relation", "B")],
        RefinementOptions(
            transitive_closure=False,
            apply_case32=False,
        ),
    )
    assert unchecked.triples == [("A", "unverified_relation", "B")]


def test_the_run_parser_selects_a_refiner():
    from kroma.cli import build_run_parser

    args = build_run_parser().parse_args(["--method_config", "kroma", "--llm", "gpt-4o-mini", "--refiner", "keep_all"])
    assert args.refiner == "keep_all"
    assert build_run_parser().parse_args(["--method_config", "kroma", "--llm", "gpt-4o-mini"]).refiner == "bisimulation"
