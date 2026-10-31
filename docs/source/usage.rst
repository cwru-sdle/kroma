Usage
=====

Running an experiment
---------------------

A run reads its configs from ``experiments/configs/`` in the working
directory, so run it from a checkout of the repository (or a folder with the
same layout):

.. code-block:: bash

   kroma run \
     --method_config kroma_scibert_envo_sweet \
     --llm meta-llama/Llama-3.3-70B-Instruct-Turbo-Free \
     --baseline

=======================  ==============================================================
Flag                     Effect
=======================  ==============================================================
``--method_config``      Method config name, without ``.jsonl`` (required)
``--llm``                Model folder under ``experiments/configs/method/`` (required)
``--size``               Fraction of pairs to score: ``xsmall`` (20%), ``small``,
                         ``medium``, ``large`` or ``full`` (default)
``--reasoning``          Ask the model to reason before it answers
``--debate``             Decide each pair by a multi-agent debate
``--active_learning``    Accept an answer only when a confidence-based score is positive
``--baseline``           Write results under ``results/baseline/``
=======================  ==============================================================

A run skips pairs it has already scored, so an interrupted run resumes where
it stopped.

From Python
-----------

:class:`~kroma.pipeline.MatchingRun` takes the same arguments and returns the
metrics dictionary:

.. code-block:: python

   from kroma.cli import build_run_parser
   from kroma.pipeline import MatchingRun

   args = build_run_parser().parse_args([
       "--method_config", "kroma_scibert_envo_sweet",
       "--llm", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
       "--size", "xsmall",
   ])
   metrics = MatchingRun(args).execute()
   print(metrics["metrics"]["f1"], metrics["cost"]["cost_usd"])

The building blocks can also be used on their own, for example the metrics
and the refinement algorithms:

.. code-block:: python

   from kroma.algorithms.refinement import offline_refine
   from kroma.algorithms.utils import compute_node_ranks
   from kroma.metrics.scoring import calculate_metrics

   graph = {"disease": {"cancer", "infection"}, "cancer": {"carcinoma"}}
   compressed = offline_refine(graph, compute_node_ranks(graph))
   precision, recall, f1 = calculate_metrics([1, 0, 1], [1, 1, 1])

Datasets
--------

Place the alignment CSVs and dictionary caches under ``experiments/dataset/``.
Each track's dataset config names its CSV folder, and
``experiments/configs/dictionary.json`` names each dictionary cache.

=================  =================  =====================================
Track              Config             CSV folder
=================  =================  =====================================
Mouse-Human        ``mouse_human``    ``experiments/dataset/OAEI/Anatomy``
NCIT-DOID          ``ncit_doid``      ``experiments/dataset/OAEI/BioLLM``
Nell-DBpedia       ``nell_dbpedia``   ``experiments/dataset/OAEI/CommonKG``
YAGO-Wikidata      ``yago_wikidata``  ``experiments/dataset/OAEI/CommonKG``
ENVO-SWEET         ``envo_sweet``     ``experiments/dataset/OAEI/Biodiv``
MI-MatOnto         ``mi_matonto``     ``experiments/dataset/OAEI/MSE``
=================  =================  =====================================

Outputs
-------

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Path
     - Contents
   * - ``results/baseline/<track>/<model>_<track>.jsonl``
     - One prediction per line
   * - ``results/baseline/<track>/<model>_<track>.metrics.json``
     - Precision, recall, F1, tokens, API calls, cost and refinement timings
   * - ``reviews/baseline/<track>/expert_queries.csv``
     - Pairs sent for expert review
   * - ``logs/kroma.log``
     - Run log

Cost
----

Each metrics file has a ``cost`` block from :func:`kroma.costs.api_cost`:
API calls, tokens per call and dollars at list price for hosted models. Add
or override prices with the ``KROMA_PRICES`` environment variable, a JSON
object of dollars per million tokens:

.. code-block:: bash

   export KROMA_PRICES='{"my-model": [0.5, 1.5]}'
