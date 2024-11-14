import logging

from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingModel:
    def __init__(self):
        """Initialize the SentenceTransformer model."""
        try:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            self.model = SentenceTransformer(
                settings.EMBEDDING_MODEL, trust_remote_code=True
            )
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            if self.embedding_dim != settings.EMBEDDING_DIM:
                logger.warning(
                    f"Configured embedding dimension ({settings.EMBEDDING_DIM}) does not match model's dimension ({self.embedding_dim})."
                )
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise e

    def encode(self, texts: list) -> list:
        """Generate embeddings for a list of sentences."""
        try:
            embeddings = self.model.encode(texts, show_progress_bar=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error during embedding generation: {e}")
            raise e
