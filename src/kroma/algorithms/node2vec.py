"""Structural embeddings with node2vec, combined with text embeddings."""

import networkx as nx
from node2vec import Node2Vec
import numpy as np
import logging

logger = logging.getLogger(__name__)

def build_graph(adj_dict):
    """Build a directed :class:`networkx.DiGraph` from a parent-to-children map."""
    G = nx.DiGraph()
    for u, children in adj_dict.items():
        for v in children:
            G.add_edge(u, v)
    return G


def train_node2vec(
    graph,
    dimensions=128,
    walk_length=30,
    num_walks=200,
    p=1,
    q=1,
    workers=4,
    seed=42,
    window=10,
    min_count=1
):
    """Fit node2vec on ``graph``.

    Args:
        graph (networkx.Graph): Graph to embed.
        dimensions (int): Embedding size.
        walk_length (int): Nodes per random walk.
        num_walks (int): Walks per node.
        p (float): Return parameter.
        q (float): In-out parameter.
        workers (int): Parallel walkers.
        seed (int): Random seed.
        window (int): Word2vec context window.
        min_count (int): Word2vec minimum count.

    Returns:
        gensim.models.Word2Vec: The fitted model; vectors are in ``model.wv``
        keyed by ``str(node)``.
    """
    logger.info("Training node2vec model with %d dimensions", dimensions)
    node2vec = Node2Vec(
        graph,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=workers,
        seed=seed
    )
    model = node2vec.fit(window=window, min_count=min_count)
    return model


def get_structural_embeddings(model, node_keys):
    """Stack the node2vec vectors of ``node_keys``.

    Nodes missing from the model get a zero vector and a warning.

    Returns:
        numpy.ndarray: Array of shape ``(len(node_keys), dimensions)``.
    """
    vecs = []
    for k in node_keys:
        try:
            vecs.append(model.wv[str(k)])
        except KeyError:
            logger.warning("Node %s not found in node2vec vocabulary, using zeros", k)
            vecs.append(np.zeros(model.vector_size))
    return np.vstack(vecs)


def combine_embeddings(text_embs, struct_embs, method="concat", alpha=0.5):
    """Combine text and structural embeddings row by row.

    Args:
        text_embs (numpy.ndarray): Text vectors, one row per node.
        struct_embs (numpy.ndarray): Structural vectors, same row order.
        method (str): ``"concat"`` to place them side by side, or
            ``"weighted_sum"`` for ``alpha * text + (1 - alpha) * struct``
            (the widths must match).
        alpha (float): Text weight for ``"weighted_sum"``.

    Returns:
        numpy.ndarray: Combined vectors.

    Raises:
        ValueError: If ``method`` is unknown.
    """
    if method == "concat":
        return np.hstack([text_embs, struct_embs])
    elif method == "weighted_sum":
        return alpha * text_embs + (1 - alpha) * struct_embs
    else:
        raise ValueError(f"Unknown combine method: {method}")


def compute_combined_embeddings(
    adj_dict,
    node_keys,
    text_embs,
    n2v_kwargs=None,
    combine_method="concat",
    alpha=0.5
):
    """Embed ``node_keys`` by text and by position in the graph.

    Args:
        adj_dict (dict): Parent-to-children map of the merged ontology.
        node_keys (list): Nodes to embed, in output row order.
        text_embs (array-like): Text vectors for ``node_keys``.
        n2v_kwargs (dict | None): Keyword arguments for :func:`train_node2vec`.
        combine_method (str): See :func:`combine_embeddings`.
        alpha (float): See :func:`combine_embeddings`.

    Returns:
        numpy.ndarray: One combined vector per node.
    """
    if n2v_kwargs is None:
        n2v_kwargs = {}

    graph = build_graph(adj_dict)
    model = train_node2vec(graph, **n2v_kwargs)
    struct_embs = get_structural_embeddings(model, node_keys)
    combined = combine_embeddings(text_embs, struct_embs, method=combine_method, alpha=alpha)
    return combined
