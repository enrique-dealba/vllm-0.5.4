import logging
import time
import uuid
from datetime import datetime, timezone

import pandas as pd
import psycopg2
import pytest

from app.backend.vector_store import VectorStore
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Register the persistence marker
def pytest_configure(config):
    config.addinivalue_line("markers", "persistence: mark test as a persistence test")


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
            "embedding": None,
        }
    )


def wait_for_db(max_retries: int = 30, delay: int = 2) -> bool:
    """Wait for database to become available"""
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(settings.TIMESCALE_SERVICE_URL)
            conn.close()
            logger.info("Successfully connected to database")
            return True
        except psycopg2.OperationalError:
            logger.info(f"Database not ready, attempt {attempt + 1}/{max_retries}")
            time.sleep(delay)
    return False


class TestVectorStore:
    """Test class for VectorStore persistence"""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test"""
        self.vector_store = VectorStore()
        self.vector_store.create_tables()
        self.vector_store.create_index()
        yield
        try:
            # Ensure connection is active before cleanup
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

        # Verify initial insertion
        initial_results = self.vector_store.search(
            "unique identifier string",
            limit=10,
            metadata_filter={"source": "persistence_test"},
        )
        assert len(initial_results) == 2, "Should have found both test documents"

        initial_data = initial_results.to_dict("records")

        # Force reconnection
        self.force_reconnect()

        # Verify data after reconnection
        post_reconnect_results = self.vector_store.search(
            "unique identifier string",
            limit=10,
            metadata_filter={"source": "persistence_test"},
        )

        post_reconnect_data = post_reconnect_results.to_dict("records")

        assert len(post_reconnect_data) == len(initial_data)
        for pre, post in zip(initial_data, post_reconnect_data):
            assert pre["content"] == post["content"]
            assert pre["metadata"] == post["metadata"]
            assert pre["embedding"] == post["embedding"]
            assert abs(pre["similarity"] - post["similarity"]) < 1e-6

    def test_long_term_persistence(self, test_data):
        """Test data persistence with multiple connection cycles"""
        logger.info("Starting long-term persistence test")

        # Insert initial test data
        test_data["embedding"] = [
            self.vector_store.get_embedding(text) for text in test_data["content"]
        ]
        self.vector_store.upsert(test_data)

        initial_results = self.vector_store.search(
            "unique identifier string",
            limit=10,
            metadata_filter={"source": "persistence_test"},
        )
        initial_count = len(initial_results)
        assert initial_count > 0

        # Multiple reconnection cycles
        for cycle in range(3):
            logger.info(f"Testing persistence cycle {cycle + 1}")
            self.force_reconnect()

            cycle_results = self.vector_store.search(
                "unique identifier string",
                limit=10,
                metadata_filter={"source": "persistence_test"},
            )

            assert len(cycle_results) == initial_count

            for idx, (initial_row, cycle_row) in enumerate(
                zip(initial_results.iterrows(), cycle_results.iterrows())
            ):
                assert initial_row[1]["content"] == cycle_row[1]["content"]
                assert initial_row[1]["metadata"] == cycle_row[1]["metadata"]

        logger.info("Long-term persistence test completed successfully")

    @pytest.mark.persistence
    def test_db_persistence(self):  # Added self parameter
        """Test database persistence across container restarts"""
        logger.info("Starting container persistence test")

        # Create test record
        test_id = str(uuid.uuid4())
        test_content = f"Persistence test content {test_id}"

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

        # Verify initial insertion
        initial_results = self.vector_store.search(test_content, limit=1)
        assert len(initial_results) == 1, "Failed to insert test data"
        assert initial_results.iloc[0]["id"] == test_id, "Retrieved wrong record"

        # Force reconnection
        self.force_reconnect()

        # Verify data after reconnection
        post_restart_results = self.vector_store.search(test_content, limit=1)
        assert len(post_restart_results) == 1, "Data not persisted after reconnection"
        assert (
            post_restart_results.iloc[0]["id"] == test_id
        ), "Retrieved wrong record after reconnection"

        logger.info("Container persistence test completed successfully")
