Matching two ontologies
=======================

:func:`kroma.match` takes two ontologies and returns the classes that refer to
the same thing. It reads any RDF format rdflib understands: RDF/XML
(``.owl``), Turtle, N-Triples, JSON-LD and more.

.. code-block:: bash

   pip install kroma-om openai
   export OPENAI_API_KEY=sk-...

From Python
-----------

.. code-block:: python

   import kroma

   result = kroma.match("mouse.owl", "human.owl", model="gpt-4o-mini")

   for a in result[:3]:
       print(a.source_label, "=", a.target_label, a.confidence, a.method)

   result.save("mouse-human.csv")   # or .tsv, or .rdf for the OAEI format
   print(result.llm_calls, result.input_tokens, result.output_tokens)

Each :class:`~kroma.alignment.Alignment` has the two IRIs and labels, the
model's confidence (0-10), the retrieval similarity, and ``method``:
``"exact"`` when a label or synonym is identical, ``"llm"`` when the model
decided.

From the command line
---------------------

.. code-block:: bash

   kroma match mouse.owl human.owl -o mouse-human.csv --model gpt-4o-mini

   # a quick trial on the first 50 classes
   kroma match mouse.owl human.owl --limit 50 -o trial.csv

Run ``kroma match --help`` for every option.

How it works
------------

For each source class, KROMA:

1. **Retrieves** the ``top_k`` most similar target classes, comparing all
   labels and synonyms of both (:func:`~kroma.matcher.find_candidates`).
2. **Accepts exact names** without a model call, after normalising case,
   spacing and punctuation.
3. **Asks the language model** about the remaining candidates, most similar
   first, showing it both classes' labels, synonyms, parents and children
   (:func:`~kroma.matcher.build_prompt`). It keeps the first Yes.

On the OAEI Anatomy track (mouse, 2,743 classes, against human, 3,304
classes), loading takes about a second, retrieval under a second, and:

==========================================  ===============
Measure                                     Value
==========================================  ===============
Reference target in the top 1 / 3 / 10      81% / 89% / 94%
Exact-name matches that are correct         99%
Reference pairs settled by exact names      62%
==========================================  ===============

So most of the reference is found without a model call, and the model judges
at most ``top_k`` candidates for each remaining class.

Choosing the options
--------------------

``top_k``
   More candidates raise recall but cost more calls for classes with no
   match. ``3`` is the default; ``5`` to ``10`` finds more.

``embedder``
   How candidates are retrieved. The default ``"lexical"`` compares character
   n-grams of all labels and synonyms: it needs no ML install and takes under
   a second. A sentence-embedding model finds a few more matches, mostly
   pairs whose names differ:

   .. code-block:: python

      kroma.match(src, tgt, embedder="sentence-transformers:sentence-transformers/all-MiniLM-L6-v2")

   This needs ``pip install "kroma-om[models]"``. You can also pass any
   object with an ``encode(list[str])`` method. Use a model trained for
   sentence similarity: a plain language model such as SciBERT retrieves far
   worse than the default. On OAEI Anatomy, the reference target was in the
   top 3:

   .. list-table::
      :header-rows: 1

      * - Embedder
        - All pairs
        - Pairs with no shared name
      * - all-MiniLM-L6-v2
        - 92%
        - 78%
      * - ``lexical`` (default)
        - 89%
        - 71%
      * - SciBERT (not similarity-trained)
        - 54%
        - 26%

``min_confidence``
   Keep only answers at this confidence or above, trading recall for
   precision. Every alignment records its confidence, so you can also filter
   afterwards.

``first_match_only``
   ``True`` (default) keeps at most one target per source class. Set it to
   ``False`` to judge every candidate and keep every Yes.

``workers``
   Source classes judged in parallel (default 4). Lower it if your provider
   rate-limits you; failed calls are retried with backoff.

Choosing the model
------------------

.. code-block:: python

   kroma.match(src, tgt, backend="togetherai", model="meta-llama/Llama-3.3-70B-Instruct-Turbo")

   # any function that takes (system_prompt, user_prompt) and returns text
   kroma.match(src, tgt, llm=lambda system, user: my_client.ask(system, user))

The model must answer in ``<answer>Yes</answer>`` or ``<answer>No</answer>``
tags with ``<confidence>0-10</confidence>``; the prompt asks for this. To add
a provider for the ``backend`` option, see :doc:`extending`.

Relation to the paper
---------------------

:func:`kroma.match` is the retrieval-and-judging core of KROMA as a library
call. The paper's benchmark runs (``kroma run``, see :doc:`usage`) add dataset
configs, cached dictionaries, SPARQL and ConceptNet context, multi-agent
debate, active learning and graph refinement on top.
