from sentence_transformers import SentenceTransformer

from app.config import settings


class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer(
            settings.EMBEDDING_MODEL, trust_remote_code=True
        )
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list) -> list:
        """Generate embeddings for a list of sentences."""
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()
