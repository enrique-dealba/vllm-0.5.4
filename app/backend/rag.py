import logging
from typing import Any, Dict

import pandas as pd

from app.backend.vector_store import VectorStore
from app.llm_logic import generate

logger = logging.getLogger(__name__)


class RAG:
    def __init__(self):
        """Initialize the RAG system with VectorStore."""
        self.vector_store = VectorStore()

    def get_relevant_context(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """Retrieve relevant context from the vector store based on the query."""
        try:
            results = self.vector_store.search(query, limit=top_k)
            return results
        except Exception as e:
            logger.error(f"Error retrieving relevant context: {e}")
            raise e

    def generate_answer(self, question: str) -> Dict[str, Any]:
        """Generate an answer based on the retrieved context."""
        try:
            context = self.get_relevant_context(question)
            # Convert DataFrame context to string representation
            context_str = context.to_string(index=False)

            query = f"""Using the following background context:
{context_str}
Please respond to the following question:
{question}
Provide a clear and concise answer based only on the context provided above. Keep your writing under 100 words."""

            response, execution_time = generate(query)  # Using vanilla generate here
            return {
                "response": response,
                "execution_time_seconds": execution_time,
                "context": context.to_dict(orient="records"),
            }
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise e
