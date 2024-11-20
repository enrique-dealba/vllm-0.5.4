import logging
import time
from datetime import datetime, timezone

import pandas as pd
import psycopg2
import pytest

from app.backend.vector_store import VectorStore
from app.config import settings

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
