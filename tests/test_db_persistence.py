import os
import time
import uuid
from typing import Generator

import psycopg2
import pytest

# Constants
DB_NAME = os.getenv("POSTGRES_DB", "postgres")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "test_db")
DB_PORT = os.getenv("DB_PORT", "5432")


@pytest.fixture(scope="session")
def db_connection() -> Generator[psycopg2.extensions.connection, None, None]:
    """Create a database connection fixture."""
    retries = 5
    while retries > 0:
        try:
            conn = psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT,
            )
            conn.autocommit = True
            yield conn
            conn.close()
            return
        except psycopg2.OperationalError:
            retries -= 1
            if retries == 0:
                raise
            time.sleep(2)


@pytest.fixture(scope="session")
def insert_test_record(db_connection) -> str:
    """Insert a test record and return its content."""
    unique_content = f"Test persistence {uuid.uuid4()}"
    with db_connection.cursor() as cur:
        cur.execute(
            """
            INSERT INTO embeddings (id, metadata, content, embedding)
            VALUES (gen_random_uuid(), %s, %s, array_fill(0.1, ARRAY[384]))
            """,
            ({"source": "test"}, unique_content),
        )
    return unique_content


def get_record_count(db_connection, content: str) -> int:
    """Get count of records matching content."""
    with db_connection.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM embeddings WHERE content = %s", (content,))
        return cur.fetchone()[0]


def test_database_persistence(db_connection, insert_test_record):
    """Test data persistence."""
    content = insert_test_record

    # Verify initial record
    count_before = get_record_count(db_connection, content)
    assert count_before == 1, f"Expected 1 record, found {count_before}"

    # Force new connection to simulate restart
    db_connection.close()
    new_conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
    )
    new_conn.autocommit = True

    # Verify record after reconnection
    count_after = get_record_count(new_conn, content)
    assert (
        count_after == 1
    ), f"Expected 1 record after reconnection, found {count_after}"

    new_conn.close()
