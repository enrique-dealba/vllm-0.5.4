import logging
import subprocess
import time
from datetime import datetime, timezone

import pandas as pd
import pytest
from psycopg2.errors import OperationalError

from app.backend.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for docker-compose commands
DOCKER_COMPOSE_FILE = "docker-compose.test.yml"
DOCKER_COMPOSE_CMD = ["docker", "compose", "-f", DOCKER_COMPOSE_FILE]


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


def docker_compose_cmd(cmd: list) -> None:
    """Execute docker-compose command and log output"""
    try:
        result = subprocess.run(
            DOCKER_COMPOSE_CMD + cmd, check=True, capture_output=True, text=True
        )
        logger.info(f"Docker command output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Docker command failed: {e.stderr}")
        raise


def wait_for_db(max_retries: int = 30, delay: int = 2) -> bool:
    """Wait for database to become available"""
    for attempt in range(max_retries):
        try:
            VectorStore()
            logger.info("Successfully connected to database")
            return True
        except OperationalError:
            logger.info(f"Database not ready, attempt {attempt + 1}/{max_retries}")
            time.sleep(delay)
    return False


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
    """Test data persistence across database restarts"""
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

    # Step 2: Stop the database
    logger.info("Stopping database container")
    docker_compose_cmd(["stop", "test_db"])
    time.sleep(5)  # Give time for container to stop

    # Step 3: Start the database
    logger.info("Starting database container")
    docker_compose_cmd(["start", "test_db"])

    # Wait for database to be ready
    assert wait_for_db(), "Database failed to become ready after restart"

    # Step 4: Create new connection and verify data
    logger.info("Creating new connection and verifying data persistence")
    new_vector_store = VectorStore()

    # Verify data after restart
    logger.info("Querying data after restart")
    post_restart_results = new_vector_store.search(
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


def test_volume_persistence(vector_store, test_data):
    """Test data persistence in docker volume"""
    # Get volume information
    logger.info("Checking volume information")
    result = subprocess.run(
        ["docker", "volume", "ls", "--format", "{{.Name}} {{.Driver}}"],
        capture_output=True,
        text=True,
    )
    assert "timescaledb_data" in result.stdout, "TimescaleDB volume not found"

    # Insert test data
    test_data["embedding"] = [
        vector_store.get_embedding(text) for text in test_data["content"]
    ]
    vector_store.upsert(test_data)

    # Verify data exists
    initial_results = vector_store.search(
        "unique identifier string",
        limit=10,
        metadata_filter={"source": "persistence_test"},
    )
    initial_count = len(initial_results)
    assert initial_count > 0, "No test data found in database"

    # Stop and remove container while preserving volume
    logger.info("Stopping and removing container while preserving volume")
    docker_compose_cmd(["down"])
    time.sleep(5)

    # Start new container with same volume
    logger.info("Starting new container with existing volume")
    docker_compose_cmd(["up", "-d", "test_db"])

    # Wait for database to be ready
    assert wait_for_db(), "Database failed to become ready after recreation"

    # Verify data in new container
    new_vector_store = VectorStore()
    post_recreation_results = new_vector_store.search(
        "unique identifier string",
        limit=10,
        metadata_filter={"source": "persistence_test"},
    )

    assert (
        len(post_recreation_results) == initial_count
    ), "Data count mismatch after container recreation"
