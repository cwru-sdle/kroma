"""Single-model and debate inference strategies."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from kroma.agents.factory import create_agents, create_agents_from_config
from kroma.agents.utils import (
    active_learning_score,
    extract_confidence,
    extract_yes_no,
    majority_vote,
)
from kroma.tasks.prompts.builders import (
    build_last_round_prompt,
    build_round_prompt,
    build_task_prompt,
)
from kroma.tasks.query_engine import query


def expand_term(term, pcmap, dictionary, options):
    """Gather a concept's context as a prompt-ready tuple.

    Args:
        term (tuple): ``(code, ...)``; only the code is used.
        pcmap (dict): Parent and child maps for the concept's ontology.
        dictionary (dict): Cached dictionaries.
        options (list[str]): Context sources for
            :func:`kroma.tasks.query_engine.query`.

    Returns:
        tuple: ``(code, parents, children, synonyms, labels)``.
    """
    parents, children, synonyms, labels = query(term[0], pcmap, dictionary, options)
    return term[0], parents, children, synonyms, labels


def _run_rounds(agents, n_rounds, src, tgt, context_block, reasoning):
    history: List[Tuple[int, int]] = []
    metrics = {"input_token": 0, "output_token": 0, "api_calls": 0}
    for round_idx in range(n_rounds):
        if round_idx == 0:
            system_prompt, base_prompt = build_task_prompt(src, tgt, reasoning=reasoning)
        elif round_idx == n_rounds - 1:
            system_prompt, base_prompt = build_last_round_prompt(src, tgt, history, perturb=False)
        else:
            system_prompt, base_prompt = build_round_prompt(src, tgt, history, perturb=False)
        user_prompt = context_block + base_prompt
        for agent in agents.values():
            response, input_tokens, output_tokens = agent.generate(
                system_prompt,
                user_prompt,
                include_tokens=True,
                truncate=False,
            )
            history.append((extract_yes_no(response), extract_confidence(response)))
            metrics["input_token"] += input_tokens
            metrics["output_token"] += output_tokens
            metrics["api_calls"] += 1
    prediction, confidence = majority_vote(history)
    return prediction, metrics, confidence


class InferenceStrategy(ABC):
    """Chooses agents and how many debate rounds to run."""

    @abstractmethod
    def agents(self, backend, model_name, agent_configs):
        """Return the agent mapping and the number of rounds.

        Returns:
            tuple[dict[int, BaseModel], int | None]: Agents and rounds;
            ``None`` rounds means use the ``n_rounds`` passed to :meth:`run`.
        """

    def run(
        self,
        backend,
        model_name,
        source_term,
        target_term,
        pcmaps,
        dictionary,
        options,
        agent_configs,
        n_rounds=3,
        reasoning=False,
        active_learning=False,
        f1_score=0.0,
        context=None,
    ):
        """Score one source/target pair.

        Every agent answers in every round; the prediction is the majority
        answer and the confidence its average (see
        :func:`kroma.agents.utils.majority_vote`).

        Args:
            backend (str): Chat backend name.
            model_name (str): Model name.
            source_term (tuple): Source ``(code, ...)``.
            target_term (tuple): Target ``(code, ...)``.
            pcmaps (dict): ``{"source": pcmap, "target": pcmap}``.
            dictionary (dict): Cached dictionaries.
            options (list[str]): Context sources.
            agent_configs (list[dict]): Agent configs for debate.
            n_rounds (int): Rounds when the strategy does not fix them.
            reasoning (bool): Passed to the prompt builder.
            active_learning (bool): Accept the answer only when
                :func:`~kroma.agents.utils.active_learning_score` is positive.
            f1_score (float): Running F1, which scales the active-learning
                noise.
            context (str | None): Text placed before every user prompt.

        Returns:
            tuple: ``(prediction, metrics, confidence, accept)``. ``prediction``
            is 1 or 0, ``metrics`` holds ``input_token``, ``output_token`` and
            ``api_calls``, ``confidence`` is 0 to 10, and ``accept`` says
            whether to refine the graph with this answer.
        """
        agents, rounds = self.agents(backend, model_name, agent_configs)
        if rounds is None:
            rounds = n_rounds
        source = expand_term(source_term, pcmaps["source"], dictionary, options)
        target = expand_term(target_term, pcmaps["target"], dictionary, options)
        prediction, metrics, confidence = _run_rounds(
            agents,
            rounds,
            source,
            target,
            context or "",
            reasoning,
        )
        accept = True
        if active_learning:
            accept = active_learning_score(confidence, f1_score) > 0
        return prediction, metrics, confidence, accept


class SingleModelInference(InferenceStrategy):
    """One model, one round."""

    def agents(self, backend, model_name, agent_configs):
        """Return one agent for ``backend``/``model_name`` and one round."""
        return create_agents(1, backend, model_name), 1


class DebateInference(InferenceStrategy):
    """One agent per config, across the configured number of rounds.

    Args:
        n_rounds (int): Debate rounds.
    """

    def __init__(self, n_rounds: int):
        self.n_rounds = n_rounds

    def agents(self, backend, model_name, agent_configs):
        """Return one agent per entry in ``agent_configs`` and ``n_rounds``."""
        return create_agents_from_config(agent_configs), self.n_rounds


def infer(
    debate: bool,
    backend: str,
    model_name: str,
    source_term: Tuple[str, str],
    target_term: Tuple[str, str],
    pcmaps: Dict[str, Dict[str, set]],
    dictionary: Dict[str, Any],
    options: List[str],
    embed_model=None,
    store=None,
    agent_configs: List[Dict[str, str]] = None,
    n_rounds: int = 3,
    dropout: float = 0.5,
    reasoning: bool = False,
    active_learning: bool = False,
    f1_score: float = 0.0,
    bisim: bool = False,
    context: str = None,
):
    """Run single-model or debate inference for one alignment pair.

    ``embed_model``, ``store``, ``dropout`` and ``bisim`` are accepted so
    existing callers keep working. Context is built by the matching run and
    passed in. See :meth:`InferenceStrategy.run` for the other arguments.

    Args:
        debate (bool): Use :class:`DebateInference` instead of
            :class:`SingleModelInference`.

    Returns:
        tuple: ``(prediction, metrics, confidence, accept)``.
    """
    strategy = DebateInference(n_rounds) if debate else SingleModelInference()
    return strategy.run(
        backend=backend,
        model_name=model_name,
        source_term=source_term,
        target_term=target_term,
        pcmaps=pcmaps,
        dictionary=dictionary,
        options=options,
        agent_configs=agent_configs or [],
        n_rounds=n_rounds,
        reasoning=reasoning,
        active_learning=active_learning,
        f1_score=f1_score,
        context=context,
    )
