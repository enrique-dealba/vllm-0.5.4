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
            time.sleep(2)
            self.vector_store.ensure_connection()
            logger.info("Database connection re-established")
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            raise
