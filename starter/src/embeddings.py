"""Real embedding provider using local SentenceTransformers."""
from typing import Protocol
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """SentenceTransformers local embedder; never falls back to random/mock vectors."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        cleaned_name = (model_name or "").strip() or DEFAULT_MODEL
        self.model_name = cleaned_name
        self._model = SentenceTransformer(self.model_name)
        if hasattr(self._model, "get_embedding_dimension"):
            dim = self._model.get_embedding_dimension()
        else:
            dim = self._model.get_sentence_embedding_dimension()
        if not dim or dim <= 0:
            raise RuntimeError(f"Could not determine embedding dimension for {self.model_name}")
        self.dimensions: int = int(dim)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [vec.tolist() for vec in embeddings]

    def embed_query(self, question: str) -> list[float]:
        if not question or not question.strip():
            raise ValueError("Query question cannot be empty")
        vec = self._model.encode(
            [question.strip()],
            show_progress_bar=False,
            normalize_embeddings=True,
        )[0]
        return vec.tolist()
