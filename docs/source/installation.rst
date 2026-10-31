Installation
============

KROMA requires Python 3.12 or newer.

From PyPI
---------

.. code-block:: bash

   pip install kroma-om            # core: loaders, refinement, metrics, CLI
   pip install "kroma-om[models]"  # plus the model backends needed for a run

The ``models`` extra installs PyTorch, Transformers, sentence-transformers,
node2vec, the OpenAI and Together clients, SPARQLWrapper, NLTK and
scikit-learn. ``kroma --help`` works without it.

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
