"""Pluggable refinement: choose how the merged graph's partition is refined.

:class:`~kroma.pipeline.MatchingRun` refines through a :class:`Refiner`. The
default, ``"bisimulation"``, is the method of the paper
(:mod:`kroma.algorithms.refinement`). Add your own with :func:`register_refiner`
and select it with ``kroma run --refiner <name>``.
"""

from typing import Any, Callable, Dict, List, Protocol, Set, Tuple

from kroma.algorithms.refinement import offline_refine, online_refine


class Refiner(Protocol):
    """How a run refines the merged graph, whole and as answers arrive."""

    def offline(self, adj: Dict[Any, Any], rank_attr: Dict[Any, int]) -> Dict[Any, Any]:
        """Refine the whole graph.

        Args:
            adj (dict): Parent to children.
            rank_attr (dict): Node rank.

        Returns:
            dict: The compressed graph.
        """
        ...

    def online(
        self,
        full_graph: Dict[Any, Set[Any]],
        rank_attr: Dict[Any, int],
        equiv_classes: Dict[Any, Any],
        delta_edges: List[Tuple[Any, Any]],
        pred: int,
    ) -> Tuple[Dict[Any, Any], List[Tuple[Any, Any]]]:
        """Update the partition after a model answers about new edges.

        Args:
            full_graph (dict): Parent to set of children.
            rank_attr (dict): Node rank.
            equiv_classes (dict): Node to its block representative.
            delta_edges (list[tuple]): The ``(u, v)`` edges the model judged.
            pred (int): The model's answer, 1 or 0.

        Returns:
            tuple: The updated partition, and the edges to ask an expert about.
        """
        ...


class BisimulationRefiner:
    """Bisimulation-based partition refinement, the method of the KROMA paper."""

    def offline(self, adj: Dict[Any, Any], rank_attr: Dict[Any, int]) -> Dict[Any, Any]:
        """Refine with :func:`~kroma.algorithms.refinement.offline_refine`.

        Args:
            adj (dict): Parent to children.
            rank_attr (dict): Node rank.

        Returns:
            dict: The compressed graph.
        """
        return offline_refine(adj, rank_attr)

    def online(
        self,
        full_graph: Dict[Any, Set[Any]],
        rank_attr: Dict[Any, int],
        equiv_classes: Dict[Any, Any],
        delta_edges: List[Tuple[Any, Any]],
        pred: int,
    ) -> Tuple[Dict[Any, Any], List[Tuple[Any, Any]]]:
        """Update with :func:`~kroma.algorithms.refinement.online_refine`.

        Args:
            full_graph (dict): Parent to set of children.
            rank_attr (dict): Node rank.
            equiv_classes (dict): Node to its block representative.
            delta_edges (list[tuple]): The ``(u, v)`` edges the model judged.
            pred (int): The model's answer, 1 or 0.

        Returns:
            tuple: The updated partition, and the edges to ask an expert about.
        """
        return online_refine(full_graph, rank_attr, equiv_classes, delta_edges, pred)


REFINERS: Dict[str, Callable[[], Refiner]] = {"bisimulation": BisimulationRefiner}


def register_refiner(name: str, factory: Callable[[], Refiner]) -> None:
    """Register a refinement method.

    Args:
        name (str): The name to select it by.
        factory (Callable[[], Refiner]): Builds the refiner.
    """
    REFINERS[name] = factory


def create_refiner(name: str = "bisimulation") -> Refiner:
    """Build a registered refiner.

    Args:
        name (str): The refiner's name.

    Returns:
        Refiner: The refiner.

    Raises:
        ValueError: When no refiner has that name.
    """
    if name not in REFINERS:
        raise ValueError(f"Unknown refiner {name!r}; available: {', '.join(sorted(REFINERS))}")
    return REFINERS[name]()
