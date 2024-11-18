import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd
import pytest

from app.backend.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mock_llm_response(text: str) -> Dict[Any, Any]:
    """Mock LLM response based on input text"""
    # Simulate different responses based on text content
    if "pokemon" in text.lower():
        return {
            "source_files": ["pokemon_data.json"],
            "json_keys_summary": ["name", "type", "abilities"],
            "descriptive_labels": {"category": "pokemon", "type": "electric"},
            "context_info": "Information about Pokemon characters",
            "num_values": 3,
            "priority_level": 2,
        }
    else:
        return {
            "source_files": ["furniture_data.json"],
            "json_keys_summary": ["item", "material", "dimensions"],
            "descriptive_labels": {"category": "furniture", "type": "home"},
            "context_info": "Information about furniture items",
            "num_values": 3,
            "priority_level": 1,
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
def sample_chunks():
    """Fixture to create sample text chunks"""
    return [
        "Pikachu is an electric-type Pokemon known for its yellow color and lightning bolt tail.",
        "Our newest collection features oak tables and leather chairs perfect for any dining room.",
    ]


def test_rag_pipeline(vector_store, sample_chunks):
    """Test the complete RAG pipeline including storage and retrieval"""
    metadata_records = []

    # Process each chunk
    for chunk in sample_chunks:
        # Mock LLM response
        llm_response = mock_llm_response(chunk)

        # Generate embedding for metadata
        metadata_embedding = vector_store.get_embedding(str(llm_response))
        chunk_id = str(uuid.uuid4())

        # Create document record
        document = {
            "id": chunk_id,
            "metadata": llm_response,
            "content": chunk,
            "embedding": metadata_embedding,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        metadata_records.append(document)

    # Test storage
    records_df = pd.DataFrame(metadata_records)
    vector_store.upsert(records_df)

    # Verify all records were stored
    results = vector_store.search("", limit=100)
    assert len(results) == 2, "Expected 2 records to be stored"

    # Test retrieval with specific query
    pokemon_query = "What is Pikachu?"
    pokemon_results = vector_store.search(pokemon_query, limit=1)
    assert len(pokemon_results) == 1, "Expected 1 result for Pokemon query"
    assert (
        "Pikachu" in pokemon_results.iloc[0]["content"]
    ), "Expected Pikachu content in top result"

    # Verify metadata structure in results
    first_result = pokemon_results.iloc[0]
    assert "metadata" in first_result, "Expected metadata in results"
    assert (
        "source_files" in first_result["metadata"]
    ), "Expected source_files in metadata"
    assert first_result["metadata"]["descriptive_labels"]["category"] == "pokemon"

    # Test retrieval with furniture query
    furniture_query = "Tell me about tables"
    furniture_results = vector_store.search(furniture_query, limit=1)
    assert len(furniture_results) == 1, "Expected 1 result for furniture query"
    assert (
        "table" in furniture_results.iloc[0]["content"].lower()
    ), "Expected furniture content in top result"

    # Test similarity scores
    pokemon_similarity = pokemon_results.iloc[0]["similarity"]
    furniture_similarity = furniture_results.iloc[0]["similarity"]
    assert isinstance(pokemon_similarity, float), "Expected float similarity score"
    assert isinstance(furniture_similarity, float), "Expected float similarity score"


def test_edge_cases(vector_store, sample_chunks):
    """Test edge cases in the RAG pipeline"""
    metadata_records = []

    # Create and store sample data
    for chunk in sample_chunks:
        llm_response = mock_llm_response(chunk)
        metadata_embedding = vector_store.get_embedding(str(llm_response))
        document = {
            "id": str(uuid.uuid4()),
            "metadata": llm_response,
            "content": chunk,
            "embedding": metadata_embedding,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        metadata_records.append(document)

    records_df = pd.DataFrame(metadata_records)
    vector_store.upsert(records_df)

    # Test empty query
    empty_results = vector_store.search("", limit=1)
    assert len(empty_results) <= 1, "Empty query should respect limit"

    # Test query with no matches
    no_match_results = vector_store.search("xyzabc123", limit=1)
    assert len(no_match_results) == 0, "Non-matching query should return empty results"

    # Test large limit
    large_limit_results = vector_store.search("pokemon", limit=1000)
    assert len(large_limit_results) == len(
        sample_chunks
    ), "Large limit should not exceed total records"


def test_metadata_filtering(vector_store, sample_chunks):
    """Test metadata filtering in the RAG pipeline"""
    metadata_records = []

    # Create and store sample data
    for chunk in sample_chunks:
        llm_response = mock_llm_response(chunk)
        metadata_embedding = vector_store.get_embedding(str(llm_response))
        document = {
            "id": str(uuid.uuid4()),
            "metadata": llm_response,
            "content": chunk,
            "embedding": metadata_embedding,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        metadata_records.append(document)

    records_df = pd.DataFrame(metadata_records)
    vector_store.upsert(records_df)

    # Test filtering by metadata
    pokemon_results = vector_store.search(
        "pokemon", limit=1, metadata_filter={"descriptive_labels.category": "pokemon"}
    )
    assert len(pokemon_results) == 1
    assert (
        pokemon_results.iloc[0]["metadata"]["descriptive_labels"]["category"]
        == "pokemon"
    )
