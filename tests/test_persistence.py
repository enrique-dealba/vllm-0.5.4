import logging
import time
import uuid
from datetime import datetime, timezone

import pandas as pd
import psycopg2
import pytest

from app.backend.vector_store import VectorStore
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "persistence: mark test as a persistence test")


@pytest.fixture(scope="session")
def db_connection():
    """Create database connection fixture."""
    conn = psycopg2.connect(settings.TIMESCALE_SERVICE_URL)
    conn.autocommit = True
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def setup_db(db_connection):
    """Setup database schema."""
    with db_connection.cursor() as cur:
        # Create extensions
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")

        # Create table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id UUID PRIMARY KEY,
                metadata JSONB,
                content TEXT,
                embedding vector(384),
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            );
        """)
    yield


@pytest.fixture(scope="module")
def test_data():
    """Create test data with consistent IDs for verification"""
    timestamp = datetime.now(timezone.utc)
    return pd.DataFrame(
        {
            "id": [str(uuid.uuid4()) for _ in range(2)],
            "metadata": [
                {
                    "source": "persistence_test",
                    "type": "document",
                    "test_id": f"test{i+1}",
                    "timestamp": timestamp.isoformat(),
                }
                for i in range(2)
            ],
            "content": [
                f"Persistence test document {i+1} - unique identifier string"
                for i in range(2)
            ],
            "embedding": None,
            "created_at": [timestamp.isoformat() for _ in range(2)],
        }
    )


class TestVectorStore:
    """Test class for VectorStore persistence"""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self, setup_db):
        """Setup and teardown for each test"""
        self.vector_store = VectorStore()
        self.vector_store.create_tables()
        self.vector_store.create_index()
        yield
        try:
            self.vector_store.ensure_connection()
            self.vector_store.delete(delete_all=True)
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    def force_reconnect(self) -> None:
        """Force database reconnection"""
        try:
            if self.vector_store.conn and not self.vector_store.conn.closed:
                self.vector_store.conn.close()
            time.sleep(2)  # Brief pause
            self.vector_store.ensure_connection()
            logger.info("Database connection re-established")
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            raise

    def test_database_persistence(self, test_data):
        """Test data persistence across database connection restarts"""
        logger.info("Starting database persistence test")

        # Step 1: Insert test data
        test_data["embedding"] = [
            self.vector_store.get_embedding(text) for text in test_data["content"]
        ]

        self.vector_store.upsert(test_data)
        logger.info("Test data inserted successfully")

        # Verify initial insertion
        initial_results = self.vector_store.search(
            "unique identifier string",
            limit=10,
            metadata_filter={"source": "persistence_test"},
        )
        assert len(initial_results) == 2, "Should have found both test documents"
        logger.info("Initial data verification successful")

        initial_data = initial_results.to_dict("records")

        # Force reconnection
        self.force_reconnect()
        logger.info("Database connection reset")

        # Verify data after reconnection
        post_reconnect_results = self.vector_store.search(
            "unique identifier string",
            limit=10,
            metadata_filter={"source": "persistence_test"},
        )

        post_reconnect_data = post_reconnect_results.to_dict("records")

        assert (
            len(post_reconnect_data) == len(initial_data)
        ), f"Data count mismatch. Expected: {len(initial_data)}, Got: {len(post_reconnect_data)}"

        for pre, post in zip(initial_data, post_reconnect_data):
            assert pre["content"] == post["content"], "Content mismatch"
            assert pre["metadata"] == post["metadata"], "Metadata mismatch"
            assert pre["embedding"] == post["embedding"], "Embedding mismatch"
            assert (
                abs(pre["similarity"] - post["similarity"]) < 1e-6
            ), "Similarity score changed significantly"

        logger.info("Post-reconnection data verification successful")

    def test_long_term_persistence(self, test_data):
        """Test data persistence with multiple connection cycles"""
        logger.info("Starting long-term persistence test")

        # Insert initial test data
        test_data["embedding"] = [
            self.vector_store.get_embedding(text) for text in test_data["content"]
        ]
        self.vector_store.upsert(test_data)
        logger.info("Initial test data inserted")

        initial_results = self.vector_store.search(
            "unique identifier string",
            limit=10,
            metadata_filter={"source": "persistence_test"},
        )
        initial_count = len(initial_results)
        assert initial_count > 0, "No initial data found"
        logger.info(f"Found {initial_count} initial records")

        # Multiple reconnection cycles
        for cycle in range(3):
            logger.info(f"Testing persistence cycle {cycle + 1}")
            self.force_reconnect()

            cycle_results = self.vector_store.search(
                "unique identifier string",
                limit=10,
                metadata_filter={"source": "persistence_test"},
            )

            assert (
                len(cycle_results) == initial_count
            ), f"Cycle {cycle + 1}: Expected {initial_count} records, found {len(cycle_results)}"

            for idx, (initial_row, cycle_row) in enumerate(
                zip(initial_results.iterrows(), cycle_results.iterrows())
            ):
                assert (
                    initial_row[1]["content"] == cycle_row[1]["content"]
                ), f"Content mismatch in record {idx} after cycle {cycle + 1}"
                assert (
                    initial_row[1]["metadata"] == cycle_row[1]["metadata"]
                ), f"Metadata mismatch in record {idx} after cycle {cycle + 1}"

            logger.info(f"Persistence cycle {cycle + 1} completed successfully")

        logger.info("Long-term persistence test completed successfully")

    @pytest.mark.persistence
    def test_db_persistence(self):
        """Test database persistence across container restarts"""
        logger.info("Starting container persistence test")

        # Create test record
        test_id = str(uuid.uuid4())
        test_content = f"Persistence test content {test_id}"
        logger.info(f"Created test record with ID: {test_id}")

        # Insert test data
        embedding = self.vector_store.get_embedding(test_content)
        test_data = pd.DataFrame(
            [
                {
                    "id": test_id,
                    "metadata": {"test_type": "persistence"},
                    "content": test_content,
                    "embedding": embedding,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            ]
        )

        self.vector_store.upsert(test_data)
        logger.info("Test data inserted successfully")

        # Verify initial insertion
        initial_results = self.vector_store.search(test_content, limit=1)
        assert len(initial_results) == 1, "Failed to insert test data"
        assert initial_results.iloc[0]["id"] == test_id, "Retrieved wrong record"
        logger.info("Initial data verification successful")

        # Force reconnection
        self.force_reconnect()
        logger.info("Database connection reset")

        # Verify data after reconnection
        post_restart_results = self.vector_store.search(test_content, limit=1)
        assert len(post_restart_results) == 1, "Data not persisted after reconnection"
        assert (
            post_restart_results.iloc[0]["id"] == test_id
        ), "Retrieved wrong record after reconnection"
        logger.info("Post-reconnection data verification successful")

        logger.info("Container persistence test completed successfully")
