"""Embedding providers backed by Hugging Face models."""

from typing import List

from .providers import EmbeddingProvider

class HFEmbeddingProvider(EmbeddingProvider):
    """Embed text with a Hugging Face encoder, using the ``[CLS]`` token vector.

    Runs on CUDA when available.

    Args:
        model_name (str): Hugging Face model id, such as
            ``"allenai/scibert_scivocab_uncased"``.
    """
    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return the ``[CLS]`` vector of each string."""
        import torch
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            last_hidden = self.model(**enc).last_hidden_state
        # mean pooling on the [CLS] token
        embeddings = last_hidden[:, 0, :].cpu().numpy()
        return embeddings.tolist()

    
class STEmbeddingProvider(EmbeddingProvider):
    """Embed text with a sentence-transformers model.

    A sentence-transformers model (such as
    ``"sentence-transformers/all-MiniLM-L6-v2"``) uses its own pooling; a
    plain Hugging Face encoder is mean-pooled. Runs on CUDA when available.

    Args:
        model_name (str): Hugging Face model id.
    """
    def __init__(self, model_name: str):
        import torch
        from sentence_transformers import SentenceTransformer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return the sentence vector of each string."""
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings.tolist()