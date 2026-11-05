Extending KROMA
===============

=========================================================  ===========================================================
To add                                                     Use
=========================================================  ===========================================================
A chat model provider                                      :func:`~kroma.inference.factory.register_chat_backend`
An embedding model provider                                :func:`~kroma.inference.factory.register_embedding_backend`
A context source, such as a dictionary or SPARQL endpoint  :func:`~kroma.tasks.query_engine.register_query_strategy`
An alignment file format                                   subclass :class:`~kroma.tasks.loader.OntologyLoader`
A judging procedure                                        subclass :class:`~kroma.agents.inference.InferenceStrategy`
A refinement method                                        :func:`~kroma.algorithms.refiners.register_refiner`
=========================================================  ===========================================================

A chat backend
--------------

Subclass :class:`~kroma.model_providers.providers.ChatProvider` and register
it. The API key is read from ``<NAME>_API_KEY``:

.. code-block:: python

   from kroma.inference.factory import register_chat_backend
   from kroma.model_providers.providers import ChatProvider, parse_chat_response

   class MyProvider(ChatProvider):
       def __init__(self, api_key):
           from openai import OpenAI
           self.client = OpenAI(api_key=api_key, base_url="http://localhost:8000/v1")

       def create_completion(self, model, messages, temperature, max_tokens, response_format=None):
           return self.client.chat.completions.create(
               model=model, messages=messages,
               temperature=temperature, max_tokens=max_tokens,
           )

       def parse_response(self, response, include_tokens):
           return parse_chat_response(response, include_tokens)

   register_chat_backend("local", MyProvider)   # reads LOCAL_API_KEY

Set ``"agent_type": "local"`` in a method config to use it.

A context source
----------------

A strategy has a ``fetch`` method returning
``(parents, children, synonyms, labels)``:

.. code-block:: python

   from kroma.tasks.query_engine import register_query_strategy

   class GlossaryStrategy:
       def __init__(self, glossary):
           self.glossary = glossary

       def fetch(self, code, pcmap, dictionary, topk=3):
           return [], [], self.glossary.get(code, [])[:topk], []

   register_query_strategy("glossary", GlossaryStrategy({"heart": ["cor"]}))

Add ``"glossary"`` to a method config's ``query_options`` to use it.

A refinement method
-------------------

A refiner has an ``offline`` and an ``online`` method, with the signatures of
:func:`~kroma.algorithms.refinement.offline_refine` and
:func:`~kroma.algorithms.refinement.online_refine`. Register it, and select it
with ``kroma run --refiner keep_all``:

.. code-block:: python

   from kroma.algorithms.refiners import BisimulationRefiner, register_refiner

   class KeepAll(BisimulationRefiner):
       def offline(self, adj, rank_attr):
           return adj

   register_refiner("keep_all", KeepAll)
