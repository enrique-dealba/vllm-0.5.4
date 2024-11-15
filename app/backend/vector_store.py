import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from app.backend.embedding import EmbeddingModel
from app.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self) -> None:
        """Initialize VectorStore with database connection and embedding model."""
        self.embedder = EmbeddingModel()
        try:
            self.conn = psycopg2.connect(settings.TIMESCALE_SERVICE_URL)
            self.conn.autocommit = True
            logger.info("Connected to database successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def create_tables(self) -> None:
        """Create necessary tables and extensions."""
        try:
            with self.conn.cursor() as cur:
                # Create extensions
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                # Create embeddings table
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {settings.VECTOR_STORE_TABLE_NAME} (
                        id UUID PRIMARY KEY,
                        metadata JSONB,
                        content TEXT,
                        embedding vector({settings.VECTOR_STORE_EMBEDDING_DIMENSIONS}),
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                # Create index on metadata for faster filtering
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{settings.VECTOR_STORE_TABLE_NAME}_metadata 
                    ON {settings.VECTOR_STORE_TABLE_NAME} USING GIN (metadata);
                """)

            logger.info(
                f"Tables and extensions created in '{settings.VECTOR_STORE_TABLE_NAME}'."
            )
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise

    def create_index(self) -> None:
        """Create the vector similarity search index."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{settings.VECTOR_STORE_TABLE_NAME}_embedding 
                    ON {settings.VECTOR_STORE_TABLE_NAME} 
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """)
            logger.info("Vector similarity index created successfully.")
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise

    def upsert(self, df: pd.DataFrame) -> None:
        """Insert or update records."""
        try:
            with self.conn.cursor() as cur:
                values = []
                for _, row in df.iterrows():
                    record_id = row.get("id") or str(uuid.uuid4())
                    values.append(
                        (record_id, row["metadata"], row["content"], row["embedding"])
                    )

                execute_values(
                    cur,
                    f"""
                    INSERT INTO {settings.VECTOR_STORE_TABLE_NAME} (id, metadata, content, embedding)
                    VALUES %s
                    ON CONFLICT (id) 
                    DO UPDATE SET 
                        metadata = EXCLUDED.metadata,
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding;
                """,
                    values,
                )

            logger.info(f"Inserted {len(df)} records.")
        except Exception as e:
            logger.error(f"Error during upsert: {e}")
            raise

    def search(
        self,
        query_text: str,
        limit: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        return_dataframe: bool = True,
    ) -> Union[List[Tuple[Any, ...]], pd.DataFrame]:
        """Perform similarity search with optional filtering."""
        try:
            query_embedding = self.get_embedding(query_text)

            query = f"""
                SELECT id, metadata, content, embedding, 
                       1 - (embedding <=> %s::vector) as similarity
                FROM {settings.VECTOR_STORE_TABLE_NAME}
                WHERE 1=1
            """
            params = [query_embedding]

            if metadata_filter:
                query += " AND metadata @> %s::jsonb"
                params.append(json.dumps(metadata_filter))

            if time_range:
                start_date, end_date = time_range
                query += " AND created_at BETWEEN %s AND %s"
                params.extend([start_date, end_date])

            query += f" ORDER BY embedding <=> %s::vector LIMIT {limit}"
            params.append(query_embedding)

            with self.conn.cursor() as cur:
                cur.execute(query, params)
                results = cur.fetchall()

            if return_dataframe:
                return self._create_dataframe_from_results(results)
            return results

        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        text = text.replace("\n", " ")
        try:
            embedding = self.embedder.encode([text])[0]
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def _create_dataframe_from_results(
        self, results: List[Tuple[Any, ...]]
    ) -> pd.DataFrame:
        """Format search results as DataFrame."""
        try:
            df = pd.DataFrame(
                results, columns=["id", "metadata", "content", "embedding", "distance"]
            )
            df = pd.concat(
                [df.drop(["metadata"], axis=1), df["metadata"].apply(pd.Series)], axis=1
            )
            return df
        except Exception as e:
            logger.error(f"Error formatting results: {e}")
            raise

    def delete(
        self,
        ids: Optional[List[str]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        delete_all: bool = False,
    ) -> None:
        """Delete records using various criteria."""
        try:
            if sum(bool(x) for x in (ids, metadata_filter, delete_all)) != 1:
                raise ValueError(
                    "Provide exactly one of: ids, metadata_filter, or delete_all"
                )

            if delete_all:
                self.vec_client.delete_all()
            elif ids:
                self.vec_client.delete_by_ids(ids)
            elif metadata_filter:
                self.vec_client.delete_by_metadata(metadata_filter)

            logger.info("Delete operation completed successfully")
        except Exception as e:
            logger.error(f"Error during delete: {e}")
            raise
