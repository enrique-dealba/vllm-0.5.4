import logging
from datetime import datetime, timezone

import pandas as pd
import pytest

from app.backend.vector_store import VectorStore
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def vector_store():
    """Fixture to create a VectorStore instance"""
    try:
        vs = VectorStore()
        vs.create_tables()
        vs.create_index()
        yield vs
        # Cleanup
        vs.delete(delete_all=True)
    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        raise


@pytest.fixture
def sample_data():
    """Fixture to create sample data"""
    return pd.DataFrame(
        {
            "id": ["test-1", "test-2"],
            "metadata": [
                {"source": "test1", "type": "document"},
                {"source": "test2", "type": "document"},
            ],
            "content": ["This is a test document 1", "This is a test document 2"],
            "created_at": [
                datetime.now(timezone.utc).isoformat(),
                datetime.now(timezone.utc).isoformat(),
            ],
        }
    )


def test_vector_store_connection(vector_store):
    """Test basic connection to vector store"""
    assert vector_store.vec_client is not None
    assert vector_store.embedder is not None


def test_embedding_generation(vector_store):
    """Test embedding generation"""
    text = "This is a test document"
    embedding = vector_store.get_embedding(text)
    assert len(embedding) == settings.EMBEDDING_DIM
    assert isinstance(embedding, list)
    assert all(isinstance(x, float) for x in embedding)


def test_upsert_and_search(vector_store, sample_data):
    """Test upserting data and searching"""
    # Upsert sample data
    vector_store.upsert(sample_data)

    # Search for similar documents
    query = "test document"
    results = vector_store.search(query, limit=2)

    assert len(results) == 2
    assert "content" in results.columns
    assert all("test document" in content.lower() for content in results["content"])


def test_delete_operations(vector_store, sample_data):
    """Test delete operations"""
    # Upsert sample data
    vector_store.upsert(sample_data)

    # Delete by ID
    vector_store.delete(ids=["test-1"])
    results = vector_store.search("test", limit=2)
    assert len(results) == 1

    # Delete all
    vector_store.delete(delete_all=True)
    results = vector_store.search("test", limit=2)
    assert len(results) == 0


def test_metadata_filtering(vector_store, sample_data):
    """Test metadata filtering in search"""
    # Upsert sample data
    vector_store.upsert(sample_data)

    # Search with metadata filter
    results = vector_store.search("test", limit=2, metadata_filter={"type": "document"})
    assert len(results) > 0

    # Search with non-matching metadata filter
    results = vector_store.search(
        "test", limit=2, metadata_filter={"type": "non-existent"}
    )
    assert len(results) == 0
