"""Fill the prompt templates for a concept pair.

A concept is passed as a tuple ``(term, parents, children, synonyms,
labels)``, as returned by :func:`kroma.agents.inference.expand_term`.
"""

from typing import List, Tuple, Any, Optional

from .templates import SYSTEM_PROMPT, TASK_PROMPT, ROUND_PROMPT, LAST_ROUND_PROMPT

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def clean_terms(terms: List[str]) -> List[str]:
    """Drop terms that are shorter than two characters or contain digits or underscores.

    Such terms are usually codes rather than names.

    Example:
        >>> clean_terms(["heart", "C123", "a", "left_ventricle"])
        ['heart']
    """
    return [t for t in terms if len(t) >= 2 and '_' not in t and not any(c.isdigit() for c in t)]


def list_to_str(items: List[Any]) -> str:
    """Join items with commas, or return ``"none"`` for an empty list.

    Example:
        >>> list_to_str(["heart", "cardiac organ"])
        'heart, cardiac organ'
        >>> list_to_str([])
        'none'
    """
    return ', '.join(map(str, items)) if items else 'none'


# ----------------------------------------------------------------------------
# Few‑shot formatting
# ----------------------------------------------------------------------------

def format_demonstrations(
    examples: List[Tuple[
        Tuple[str, List[str], List[str], List[str], List[str]],
        Tuple[str, List[str], List[str], List[str], List[str]],
        int,  # 1=Yes, 0=No
        int
    ]]
) -> str:
    """Turn few-shot examples into a prompt block.

    Args:
        examples (list[tuple]): ``(source, target, label, confidence)`` with
            ``label`` 1 for Yes and 0 for No.

    Returns:
        str: An ``Examples:`` block, one line per example.
    """
    lines = ["Examples:"]
    for src, tgt, lbl, conf in examples:
        ans = 'Yes' if lbl == 1 else 'No'
        lines.append(f"Source: {src[0]}; Target: {tgt[0]} -> {ans} (confidence {conf})")
    return '\n'.join(lines) + '\n'


# ----------------------------------------------------------------------------
# Prompt builders
# ----------------------------------------------------------------------------

def build_task_prompt(
    source: Tuple[str, List[str], List[str], List[str], List[str]],
    target: Tuple[str, List[str], List[str], List[str], List[str]],
    examples: Optional[List[Tuple[Any, Any, int, int]]] = None,
    reasoning: bool = False,
) -> Tuple[str, str]:
    """Build the system and user prompts that ask whether two concepts match.

    Each concept's own term is added to its names, and every list is passed
    through :func:`clean_terms`.

    Args:
        source (tuple): Source ``(term, parents, children, synonyms, labels)``.
        target (tuple): Target ``(term, parents, children, synonyms, labels)``.
        examples (list[tuple] | None): Few-shot examples for
            :func:`format_demonstrations`.
        reasoning (bool): Accepted for callers; the prompt does not change.

    Returns:
        tuple[str, str]: ``(system_prompt, user_prompt)``.
    """
    _, s_par, s_chi, s_syn, s_lab = source
    _, t_par, t_chi, t_syn, t_lab = target

    # prepare and clean
    s_lab = clean_terms(list(set(s_lab + [source[0]])))
    t_lab = clean_terms(list(set(t_lab + [target[0]])))
    s_par, s_chi, s_syn = map(clean_terms, (s_par, s_chi, s_syn))
    t_par, t_chi, t_syn = map(clean_terms, (t_par, t_chi, t_syn))

    demos = format_demonstrations(examples) if examples else ''
    user = TASK_PROMPT.format(
        demonstrations=demos,
        src_names=list_to_str(s_lab),
        src_parents=list_to_str(s_par),
        src_children=list_to_str(s_chi),
        src_synonyms=list_to_str(s_syn),
        tgt_names=list_to_str(t_lab),
        tgt_parents=list_to_str(t_par),
        tgt_children=list_to_str(t_chi),
        tgt_synonyms=list_to_str(t_syn),
    )
    return SYSTEM_PROMPT, user


def _history_block(history) -> str:
    """Format either an agent-id map or a list of (answer, confidence) pairs."""
    if isinstance(history, dict):
        return '\n'.join(
            f"Agent {aid+1}: {' '.join(map(str, ans))}"
            for aid, ans in history.items()
        )
    return '\n'.join(f"{ans} (confidence {conf})" for ans, conf in history or [])


def _coerce_history(source, target, history):
    """Accept build_round_prompt(history) and build_round_prompt(src, tgt, history)."""
    if history is None and isinstance(source, (dict, list)):
        return source
    return history


def build_round_prompt(
    source=None,
    target=None,
    history=None,
    perturb: bool = False,
) -> Tuple[str, str]:
    """Build the prompts for an intermediate debate round.

    Accepts ``build_round_prompt(history)`` and
    ``build_round_prompt(source, target, history)``.

    Args:
        source: Source concept tuple, or the history when called with one
            argument.
        target: Target concept tuple (unused by the template).
        history (dict | list): Agent id to answers, or a list of
            ``(answer, confidence)`` pairs.
        perturb (bool): Accepted for callers; unused.

    Returns:
        tuple[str, str]: ``(system_prompt, user_prompt)``.
    """
    history = _coerce_history(source, target, history)
    return SYSTEM_PROMPT, ROUND_PROMPT.format(agent_answers=_history_block(history))


def build_last_round_prompt(
    source=None,
    target=None,
    history=None,
    perturb: bool = False,
) -> Tuple[str, str]:
    """Build the prompts for the final debate round.

    Takes the same arguments as :func:`build_round_prompt` and asks for a
    conclusive answer.

    Returns:
        tuple[str, str]: ``(system_prompt, user_prompt)``.
    """
    history = _coerce_history(source, target, history)
    return SYSTEM_PROMPT, LAST_ROUND_PROMPT.format(agent_answers=_history_block(history))
