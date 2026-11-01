KROMA
=====

**Ontology Matching with Knowledge Retrieval and Large Language Models**

KROMA matches concepts across two ontologies. For each candidate pair it
retrieves the concepts' context, asks a large language model whether the two
refer to the same real-world entity, and refines the alignment as answers
arrive. It was published in the Research Track of ISWC 2025
(`paper <https://doi.org/10.1007/978-3-032-09527-5_34>`_,
`arXiv <https://arxiv.org/abs/2507.14032>`_) and is maintained by the
`SDLE Research Center <https://github.com/cwru-sdle>`_ at Case Western Reserve
University.

How it works
------------

1. **Load.** A track's alignment CSVs become a source graph, a target graph
   and labelled source-target pairs (:mod:`kroma.tasks.loader`).
2. **Retrieve.** Each concept is embedded with SciBERT text vectors fused with
   node2vec graph vectors (:mod:`kroma.algorithms.node2vec`). Its context comes
   from cached dictionaries, the ontology graph, ConceptNet, and SPARQL
   endpoints for NCIt, DOID and DBpedia (:mod:`kroma.tasks.query_engine`).
3. **Decide.** A language model answers Yes or No with a confidence, alone or
   as a multi-agent debate (:mod:`kroma.agents.inference`).
4. **Refine.** Accepted matches update the equivalence classes as they arrive
   (online refinement), and the whole graph is re-collapsed every five pairs
   (offline refinement). Pairs refinement cannot settle go to an expert
   (:mod:`kroma.algorithms.refinement`).

Quick start
-----------

.. code-block:: bash

   pip install kroma-om openai
   export OPENAI_API_KEY=sk-...
   kroma match mouse.owl human.owl -o alignments.csv

.. code-block:: python

   import kroma

   result = kroma.match("mouse.owl", "human.owl")
   result.save("alignments.csv")

The distribution is named ``kroma-om``; the Python package is ``kroma``. See
:doc:`matching` for the options.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   matching
   usage
   extending

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api

Citation
--------

.. code-block:: bibtex

   @inproceedings{nguyen_2025_kroma,
       title={KROMA: Ontology Matching with Knowledge Retrieval and Large Language Models},
       author={Lam Nguyen and Erika Barcelos and Roger French and Yinghui Wu},
       booktitle={The Semantic Web -- ISWC 2025: 24th International Semantic Web Conference, Proceedings, Part I},
       series={Lecture Notes in Computer Science},
       publisher={Springer},
       pages={629--649},
       year={2025},
       doi={10.1007/978-3-032-09527-5_34},
   }
