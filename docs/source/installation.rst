Installation
============

KROMA requires Python 3.12 or newer.

From PyPI
---------

.. code-block:: bash

   pip install kroma-om openai     # kroma.match / kroma match with OpenAI
   pip install kroma-om together   # ... or with Together AI
   pip install "kroma-om[models]"  # everything, including benchmark runs

The core package needs no machine-learning framework: :func:`kroma.match`
with the default lexical retrieval only needs the client of the chat backend
you use. The ``models`` extra adds PyTorch, Transformers,
sentence-transformers, node2vec, both chat clients, SPARQLWrapper, NLTK and
scikit-learn, for neural embedders and ``kroma run``.

From source
-----------

.. code-block:: bash

   git clone https://github.com/cwru-sdle/kroma.git
   cd kroma
   ./setup.sh                      # creates .venv with uv and installs .[dev]
   source .venv/bin/activate
   uv pip install ".[models]"

API keys
--------

Chat backends read their key from ``<BACKEND>_API_KEY``. Put them in a
``.env`` file in the working directory, which KROMA loads on import:

.. code-block:: bash

   cp .env.example .env

=================  =====================
Backend            Variable
=================  =====================
``togetherai``     ``TOGETHERAI_API_KEY``
``openai``         ``OPENAI_API_KEY``
=================  =====================

Building these docs
-------------------

.. code-block:: bash

   pip install -e ".[docs]"
   make -C docs html               # output in docs/build/html
