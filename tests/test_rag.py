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
    logger.info(f"Generating mock response for text: {text}")

    if "pokemon" in text.lower():
        response = {
            "source_files": ["pokemon_data.json"],
            "json_keys_summary": ["name", "type", "abilities"],
            "descriptive_labels": {"category": "pokemon", "type": "electric"},
            "context_info": "Information about Pokemon characters",
            "num_values": 3,
            "priority_level": 2,
        }
        logger.info(f"Generated Pokemon response: {response}")
        return response
    else:
        response = {
            "source_files": ["furniture_data.json"],
            "json_keys_summary": ["item", "material", "dimensions"],
            "descriptive_labels": {"category": "furniture", "type": "home"},
            "context_info": "Information about furniture items",
            "num_values": 3,
            "priority_level": 1,
        }
        logger.info(f"Generated furniture response: {response}")
        return response


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
        # Generate chunk embedding first
        chunk_embedding = vector_store.get_embedding(chunk)
        llm_response = mock_llm_response(chunk)

        document = {
            "id": str(uuid.uuid4()),
            "metadata": llm_response,
            "content": chunk,
            "embedding": chunk_embedding,  # Use chunk embedding instead of metadata embedding
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        metadata_records.append(document)

    # Test storage
    records_df = pd.DataFrame(metadata_records)
    vector_store.upsert(records_df)

    # Verify records exist using verify_record_exists
    for chunk in sample_chunks:
        assert vector_store.verify_record_exists(
            chunk
        ), f"Record with content '{chunk}' not found"

    # Test retrieval with specific query
    pokemon_query = "Pikachu pokemon"
    pokemon_results = vector_store.search(
        pokemon_query, limit=1, similarity_threshold=0.1
    )
    assert len(pokemon_results) == 1, "Expected 1 result for Pokemon query"
    assert "Pikachu" in pokemon_results.iloc[0]["content"]


def test_edge_cases(vector_store, sample_chunks):
    """Test edge cases in the RAG pipeline"""
    metadata_records = []

    for chunk in sample_chunks:
        chunk_embedding = vector_store.get_embedding(chunk)
        llm_response = mock_llm_response(chunk)
        document = {
            "id": str(uuid.uuid4()),
            "metadata": llm_response,
            "content": chunk,
            "embedding": chunk_embedding,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        metadata_records.append(document)

    records_df = pd.DataFrame(metadata_records)
    vector_store.upsert(records_df)

    # Test with high similarity threshold for no matches
    no_match_results = vector_store.search(
        "xyzabc123", limit=1, similarity_threshold=0.9
    )
    assert len(no_match_results) == 0, "Non-matching query should return empty results"


def test_metadata_filtering(vector_store, sample_chunks):
    """Test metadata filtering in the RAG pipeline"""
    metadata_records = []

    for chunk in sample_chunks:
        chunk_embedding = vector_store.get_embedding(chunk)
        llm_response = mock_llm_response(chunk)

        # Let's log the metadata to verify its structure
        logger.info(f"Generated metadata for chunk '{chunk}': {llm_response}")

        document = {
            "id": str(uuid.uuid4()),
            "metadata": llm_response,
            "content": chunk,
            "embedding": chunk_embedding,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        metadata_records.append(document)

    records_df = pd.DataFrame(metadata_records)
    vector_store.upsert(records_df)

    # First, verify records were stored properly
    all_results = vector_store.search("", limit=100, similarity_threshold=-1.0)
    logger.info(f"Total records found: {len(all_results)}")
    if not all_results.empty:
        for idx, row in all_results.iterrows():
            logger.info(f"Record {idx} metadata: {row['metadata']}")

    # Try searching with a very low similarity threshold first
    pokemon_results = vector_store.search(
        "pokemon",
        limit=1,
        similarity_threshold=0.0,  # Set to 0 to ensure we get results
        metadata_filter={
            "descriptive_labels": {"category": "pokemon"}
        },  # Match nested structure
    )

    # Debug output
    logger.info(f"Pokemon search results: {len(pokemon_results)}")
    if not pokemon_results.empty:
        logger.info(f"First result metadata: {pokemon_results.iloc[0]['metadata']}")

    assert len(pokemon_results) > 0, "Expected at least one Pokemon result"
    if len(pokemon_results) > 0:
        metadata = pokemon_results.iloc[0]["metadata"]
        assert metadata["descriptive_labels"]["category"] == "pokemon"
