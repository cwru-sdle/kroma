"""Parse model answers and combine them."""

import re
import random
from collections import defaultdict
from typing import List, Tuple, Dict

def extract_yes_no(response: str) -> int:
    """Parse ``<answer>Yes</answer>`` or ``<answer>No</answer>`` into 1 or 0.

    Anything else, including a missing tag, counts as No.

    Example:
        >>> extract_yes_no("<answer> yes </answer>")
        1
    """
    m = re.search(r'<answer>\s*(yes|no)\s*</answer>', response, re.IGNORECASE)
    return 1 if m and m.group(1).lower() == 'yes' else 0

def extract_confidence(response: str) -> int:
    """Parse ``<confidence>N</confidence>`` into an integer from 0 to 10.

    A missing tag or a value above 10 gives 0.

    Example:
        >>> extract_confidence("<confidence>8</confidence>")
        8
    """
    m = re.search(r'<confidence>\s*(\d{1,2})\s*</confidence>', response)
    val = int(m.group(1)) if m else 0
    return val if 0 <= val <= 10 else 0

def majority_vote(answers: List[Tuple[int,int]]) -> Tuple[int,int]:
    """Return the most common answer and the average confidence behind it.

    Args:
        answers (list[tuple[int, int]]): ``(answer, confidence)`` pairs.

    Returns:
        tuple[int, int]: Majority answer and its truncated mean confidence.
        On a tie, the answer seen first wins.

    Example:
        >>> majority_vote([(1, 8), (0, 3), (1, 6)])
        (1, 7)
    """
    counts: Dict[int, List[int]] = defaultdict(list)
    for ans, conf in answers:
        counts[ans].append(conf)
    # pick ans with most votes
    maj = max(counts, key=lambda k: len(counts[k]))
    avg_conf = int(sum(counts[maj]) / len(counts[maj]))
    return maj, avg_conf

def active_learning_score(
    confidence: int,
    f1_score: float,
    gamma: float = 1.0,
    delta: float = 1.0,
    epsilon: float = 0.1
) -> float:
    """Score whether to accept an answer during active learning.

    The score is ``(gamma + delta) * confidence - 11 * delta`` plus Gaussian
    noise scaled by ``epsilon * f1_score``. With the defaults an answer is
    accepted (score above zero) at confidence 6 or more, give or take the
    noise.

    Args:
        confidence (int): Model confidence, 0 to 10.
        f1_score (float): Running F1 of the run.
        gamma (float): Weight on confidence.
        delta (float): Weight on confidence and on the threshold.
        epsilon (float): Noise scale.

    Returns:
        float: Acceptance score; positive means accept.
    """
    noise = random.gauss(0,1) * epsilon * f1_score
    return (gamma + delta) * confidence - 11 * delta + noise
