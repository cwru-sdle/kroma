"""Graph helpers for ranking and merging ontology hierarchies."""

from collections import defaultdict, deque
from typing import Dict, Any, Set

def compute_node_ranks(
    graph: Dict[Any, Set[Any]]
) -> Dict[Any, int]:
    """Compute each node's rank: its longest distance from a root.

    Roots (no incoming edges) have rank 0. Nodes on a cycle are never
    reached and are left out.

    Args:
        graph (dict): DAG as parent to children.

    Returns:
        dict: Node to rank.

    Example:
        >>> sorted(compute_node_ranks({"a": {"b", "c"}, "b": {"c"}}).items())
        [('a', 0), ('b', 1), ('c', 2)]
    """
    indegree = defaultdict(int)
    for parent, children in graph.items():
        indegree[parent]  # ensure key exists
        for c in children:
            indegree[c] += 1

    queue = deque([n for n, deg in indegree.items() if deg == 0])
    ranks = {n: 0 for n in queue}

    while queue:
        node = queue.popleft()
        for child in graph.get(node, set()):
            ranks[child] = max(ranks.get(child, 0), ranks[node] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    return ranks


def merge_graphs(*graphs: Dict[Any, Set[Any]]) -> Dict[Any, Set[Any]]:
    """Union several parent-to-children graphs into one.

    Example:
        >>> merge_graphs({"a": {"b"}}, {"a": {"c"}}) == {"a": {"b", "c"}}
        True
    """
    merged: Dict[Any, Set[Any]] = {}
    for G in graphs:
        for u, vs in G.items():
            merged.setdefault(u, set()).update(vs)
    return merged