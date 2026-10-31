"""A small in-memory vector store with cosine-similarity search."""

import numpy as np
from typing import List, Tuple, Any


class VectorStore:
    """Store vectors under external ids and return the most similar ones.

    Example:
        >>> store = VectorStore()
        >>> store.add(["heart", "lung"], [[1.0, 0.0], [0.0, 1.0]])
        >>> store.query([0.9, 0.1], top_k=1)
        [('heart', 0.9938837289810181)]
    """

    def __init__(self):
        # list of external IDs (e.g. concept codes)
        self.ids: List[Any] = []
        # will become a (N × D) array once data is added
        self._matrix: np.ndarray = np.zeros((0, 0), dtype='float32')

    def add(self, ext_ids: List[Any], embeddings: List[List[float]]) -> None:
        """Add vectors and their ids.

        Args:
            ext_ids (list): Ids, such as concept codes, one per vector.
            embeddings (array-like): Vectors of equal width, stored as
                float32.
        """
        embs = np.vstack(embeddings).astype('float32')  # shape = (n, D)
        if self._matrix.size == 0:
            # first batch
            self._matrix = embs
        else:
            # stack on existing
            self._matrix = np.vstack([self._matrix, embs])
        self.ids.extend(ext_ids)

    def query(self, query_emb: List[float], top_k: int = 5) -> List[Tuple[Any, float]]:
        """Return the ``top_k`` ids most similar to ``query_emb``.

        Args:
            query_emb (array-like): Query vector.
            top_k (int): Number of results.

        Returns:
            list[tuple[Any, float]]: ``(id, cosine_similarity)`` pairs, most
            similar first; empty when the store is empty.
        """
        if self._matrix.size == 0:
            return []
        q = np.array(query_emb, dtype='float32').reshape(1, -1)  # (1, D)
        # normalize
        q_norm = q / np.linalg.norm(q, axis=1, keepdims=True)
        M_norm = self._matrix / np.linalg.norm(self._matrix, axis=1, keepdims=True)
        # cosine similarities
        sims = (M_norm @ q_norm.T).reshape(-1)  # shape = (N,)
        # get top_k indices
        idxs = np.argsort(-sims)[:top_k]
        return [(self.ids[i], float(sims[i])) for i in idxs]

    def get_text(self, ext_id: Any) -> str:
        """Return the text for an id; currently the id as a string."""
        return str(ext_id)
