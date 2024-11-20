import logging
import time
from datetime import datetime, timezone

import pandas as pd
import psycopg2
import pytest
from psycopg2.extensions import connection

from app.backend.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def test_data():
    """Create test data with consistent IDs for verification"""
    timestamp = datetime.now(timezone.utc)
    return pd.DataFrame(
        {
            "metadata": [
                {
                    "source": "persistence_test",
                    "type": "document",
                    "test_id": "test1",
                    "timestamp": timestamp.isoformat(),
                },
                {
                    "source": "persistence_test",
                    "type": "document",
                    "test_id": "test2",
                    "timestamp": timestamp.isoformat(),
                },
            ],
            "content": [
                "Persistence test document 1 - unique identifier string",
                "Persistence test document 2 - unique identifier string",
            ],
            "embedding": None,  # Will be generated during insertion
        }
    )


def wait_for_db(conn_string: str, max_retries: int = 30, delay: int = 2) -> bool:
    """Wait for database to become available"""
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(conn_string)
            conn.close()
            logger.info("Successfully connected to database")
            return True
        except psycopg2.OperationalError:
            logger.info(f"Database not ready, attempt {attempt + 1}/{max_retries}")
            time.sleep(delay)
    return False


def restart_db_connection(conn: connection) -> connection:
    """Close and reopen database connection"""
    try:
        if conn and not conn.closed:
            conn_params = conn.get_dsn_parameters()
            conn.close()
            logger.info("Closed existing database connection")

            # Construct connection string from parameters
            conn_string = f"postgresql://{conn_params['user']}:{conn_params.get('password', '')}@{conn_params['host']}:{conn_params['port']}/{conn_params['dbname']}"

            # Wait briefly to simulate database restart
            time.sleep(5)

            # Try to reconnect
            if wait_for_db(conn_string):
                new_conn = psycopg2.connect(conn_string)
                new_conn.autocommit = True
                logger.info("Established new database connection")
                return new_conn
            else:
                raise Exception("Failed to reconnect to database")
    except Exception as e:
        logger.error(f"Error during connection restart: {e}")
        raise
    return conn


@pytest.fixture(scope="module")
def vector_store():
    """Create VectorStore instance and clean up after tests"""
    try:
        vs = VectorStore()
        vs.create_tables()
        vs.create_index()
        yield vs
        # Cleanup after all tests
        vs.delete(delete_all=True)
    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        raise


def test_database_persistence(vector_store, test_data):
    """Test data persistence across database connection restarts"""
    logger.info("Starting database persistence test")

    # Step 1: Insert test data
    logger.info("Generating embeddings for test data")
    test_data["embedding"] = [
        vector_store.get_embedding(text) for text in test_data["content"]
    ]

    logger.info("Inserting test data")
    vector_store.upsert(test_data)

    # Verify initial insertion
    logger.info("Verifying initial data insertion")
    initial_results = vector_store.search(
        "unique identifier string",
        limit=10,
        metadata_filter={"source": "persistence_test"},
    )
    assert len(initial_results) == 2, "Should have found both test documents"

    # Store initial data for comparison
    initial_data = initial_results.to_dict("records")

    # Step 2: Restart database connection
    logger.info("Restarting database connection")
    vector_store.conn = restart_db_connection(vector_store.conn)

    # Step 3: Verify data after connection restart
    logger.info("Verifying data after connection restart")
    post_restart_results = vector_store.search(
        "unique identifier string",
        limit=10,
        metadata_filter={"source": "persistence_test"},
    )

    # Convert results to comparable format
    post_restart_data = post_restart_results.to_dict("records")

    # Compare data before and after restart
    logger.info("Comparing pre-restart and post-restart data")
    assert (
        len(post_restart_data) == len(initial_data)
    ), f"Number of records changed. Before: {len(initial_data)}, After: {len(post_restart_data)}"

    # Compare each record's critical fields
    for pre, post in zip(initial_data, post_restart_data):
        assert pre["content"] == post["content"], "Content mismatch"
        assert pre["metadata"] == post["metadata"], "Metadata mismatch"
        assert pre["embedding"] == post["embedding"], "Embedding mismatch"
        assert (
            abs(pre["similarity"] - post["similarity"]) < 1e-6
        ), "Similarity score changed significantly"


def test_long_term_persistence(vector_store, test_data):
    """Test data persistence with multiple connection cycles"""
    logger.info("Starting long-term persistence test")

    # Insert initial test data
    test_data["embedding"] = [
        vector_store.get_embedding(text) for text in test_data["content"]
    ]
    vector_store.upsert(test_data)

    # Initial verification
    initial_results = vector_store.search(
        "unique identifier string",
        limit=10,
        metadata_filter={"source": "persistence_test"},
    )
    initial_count = len(initial_results)
    assert initial_count > 0, "No test data found in database"

    # Multiple connection restart cycles
    for cycle in range(3):
        logger.info(f"Testing persistence cycle {cycle + 1}")

        # Restart connection
        vector_store.conn = restart_db_connection(vector_store.conn)

        # Verify data after restart
        cycle_results = vector_store.search(
            "unique identifier string",
            limit=10,
            metadata_filter={"source": "persistence_test"},
        )

        assert (
            len(cycle_results) == initial_count
        ), f"Data count mismatch after cycle {cycle + 1}"

        # Compare content and metadata
        for idx, (initial_row, cycle_row) in enumerate(
            zip(initial_results.iterrows(), cycle_results.iterrows())
        ):
            assert (
                initial_row[1]["content"] == cycle_row[1]["content"]
            ), f"Content mismatch in record {idx} after cycle {cycle + 1}"
            assert (
                initial_row[1]["metadata"] == cycle_row[1]["metadata"]
            ), f"Metadata mismatch in record {idx} after cycle {cycle + 1}"

    logger.info("Long-term persistence test completed successfully!")
