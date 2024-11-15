import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import psycopg2
import pytest

from app.backend.vector_store import VectorStore
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_metadata(source: str, doc_type: str) -> dict:
    """Create metadata with proper timestamp for testing."""
    return {
        "source": source,
        "type": doc_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


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
    timestamp = datetime.now(timezone.utc)
    return pd.DataFrame(
        {
            "metadata": [
                create_test_metadata("test1", "document"),
                create_test_metadata("test2", "document"),
            ],
            "content": ["This is a test document 1", "This is a test document 2"],
            "embedding": None,  # Will be generated during upsert
        }
    )


def test_vector_store_connection(vector_store):
    """Test basic connection to vector store"""
    assert vector_store.conn is not None
    assert vector_store.embedder is not None
    assert isinstance(vector_store.conn, psycopg2.extensions.connection)


def test_embedding_generation(vector_store):
    """Test embedding generation"""
    text = "This is a test document"
    embedding = vector_store.get_embedding(text)
    assert len(embedding) == settings.EMBEDDING_DIM
    assert isinstance(embedding, list)
    assert all(isinstance(x, float) for x in embedding)


def test_upsert_and_search(vector_store, sample_data):
    """Test upserting data and searching"""
    # Generate embeddings for sample data
    sample_data["embedding"] = [
        vector_store.get_embedding(text) for text in sample_data["content"]
    ]

    # Upsert sample data
    vector_store.upsert(sample_data)

    # Search for similar documents
    query = "test document"
    results = vector_store.search(query, limit=2)

    assert len(results) == 2
    assert "content" in results.columns
    assert all("test document" in content.lower() for content in results["content"])


def test_time_based_search(vector_store, sample_data):
    """Test time-based searching"""
    # Generate embeddings for sample data
    sample_data["embedding"] = [
        vector_store.get_embedding(text) for text in sample_data["content"]
    ]

    # Upsert sample data
    vector_store.upsert(sample_data)

    now = datetime.now(timezone.utc)
    hour_ago = now - timedelta(hours=1)
    hour_ahead = now + timedelta(hours=1)

    # Search within time range
    results = vector_store.search(
        "test document", limit=2, time_range=(hour_ago, hour_ahead)
    )
    assert len(results) == 2

    # Search outside time range
    results = vector_store.search(
        "test document",
        limit=2,
        time_range=(hour_ahead, hour_ahead + timedelta(hours=1)),
    )
    assert len(results) == 0


def test_delete_operations(vector_store, sample_data):
    """Test delete operations"""
    # Generate embeddings and upsert
    sample_data["embedding"] = [
        vector_store.get_embedding(text) for text in sample_data["content"]
    ]
    vector_store.upsert(sample_data)

    # Search for records
    results = vector_store.search("test", limit=2)
    assert len(results) > 0, "Search should return at least one result"

    # Get the first record's ID
    first_id = results.iloc[0]["id"]
    assert first_id is not None, "First result should have an ID"

    # Delete by ID
    vector_store.delete(ids=[first_id])
    after_delete = vector_store.search("test", limit=2)
    assert len(after_delete) == 1, "Should have one remaining record"

    # Delete all
    vector_store.delete(delete_all=True)
    final_results = vector_store.search("test", limit=2)
    assert len(final_results) == 0, "Should have no records after delete_all"


def test_metadata_filtering(vector_store, sample_data):
    """Test metadata filtering in search"""
    # Generate embeddings and upsert
    sample_data["embedding"] = [
        vector_store.get_embedding(text) for text in sample_data["content"]
    ]
    vector_store.upsert(sample_data)

    # Search with metadata filter
    results = vector_store.search("test", limit=2, metadata_filter={"type": "document"})
    assert len(results) > 0

    # Search with non-matching metadata filter
    results = vector_store.search(
        "test", limit=2, metadata_filter={"type": "non-existent"}
    )
    assert len(results) == 0


def test_combined_filtering(vector_store, sample_data):
    """Test combined metadata and time filtering"""
    # Generate embeddings and upsert
    sample_data["embedding"] = [
        vector_store.get_embedding(text) for text in sample_data["content"]
    ]
    vector_store.upsert(sample_data)

    now = datetime.now(timezone.utc)
    hour_ago = now - timedelta(hours=1)
    hour_ahead = now + timedelta(hours=1)

    # Search with both metadata and time filters
    results = vector_store.search(
        "test",
        limit=2,
        metadata_filter={"type": "document"},
        time_range=(hour_ago, hour_ahead),
    )
    assert len(results) > 0
