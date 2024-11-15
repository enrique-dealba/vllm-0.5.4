import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import psycopg2
from psycopg2.extensions import register_adapter
from psycopg2.extras import Json, execute_values

from app.backend.embedding import EmbeddingModel
from app.config import settings

logger = logging.getLogger(__name__)

# Register JSON adapter for psycopg2
register_adapter(dict, Json)


class VectorStore:
    def __init__(self) -> None:
        """Initialize VectorStore with database connection and embedding model."""
        self.embedder = EmbeddingModel()
        try:
            self.conn = psycopg2.connect(settings.TIMESCALE_SERVICE_URL)
            self.conn.autocommit = True
            logger.info("Connected to database successfully.")

            # Automatically create tables and index upon initialization
            self.create_tables()
            self.create_index()

        except Exception as e:
            logger.error(f"Failed to connect to database or initialize tables: {e}")
            raise

    def create_tables(self) -> None:
        """Create necessary tables and extensions."""
        try:
            with self.conn.cursor() as cur:
                # Create extensions
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")

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
                    # Convert embedding list to string format PostgreSQL expects
                    embedding_str = f"[{','.join(map(str, row['embedding']))}]"
                    values.append(
                        (
                            record_id,
                            Json(row["metadata"]),
                            row["content"],
                            embedding_str,
                        )
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
                        embedding = EXCLUDED.embedding::vector;
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
        similarity_threshold: float = 0.0,
    ) -> Union[List[Tuple[Any, ...]], pd.DataFrame]:
        """Perform similarity search with optional filtering."""
        try:
            query_embedding = self.get_embedding(query_text)
            embedding_str = f"[{','.join(map(str, query_embedding))}]"

            logger.debug(f"Search query text: {query_text}")
            logger.debug(f"Query embedding first few values: {query_embedding[:5]}")

            query = f"""
                SELECT 
                    id, 
                    metadata, 
                    content, 
                    embedding,
                    created_at,  -- Added
                    1 - (embedding <=> %s::vector) as similarity
                FROM {settings.VECTOR_STORE_TABLE_NAME}
                WHERE 1 - (embedding <=> %s::vector) >= %s
            """
            params = [embedding_str, embedding_str, similarity_threshold]

            if metadata_filter:
                query += " AND metadata @> %s::jsonb"
                params.append(Json(metadata_filter))

            if time_range:
                start_date, end_date = time_range
                query += " AND created_at BETWEEN %s AND %s"
                params.extend([start_date, end_date])

            query += " ORDER BY similarity DESC LIMIT %s"
            params.append(limit)

            with self.conn.cursor() as cur:
                formatted_query = cur.mogrify(query, params).decode("utf-8")
                logger.debug(f"Executing search query: {formatted_query}")

                cur.execute(query, params)
                results = cur.fetchall()

                logger.debug(f"Raw results count: {len(results)}")
                if results:
                    logger.debug(f"First result similarity: {results[0][-1]}")

                if return_dataframe:
                    df = self._create_dataframe_from_results(results)
                    if not df.empty:
                        logger.debug(f"Similarity scores: {df['similarity'].tolist()}")
                    return df
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
            if not results:
                logger.debug("No results to convert to DataFrame")
                return pd.DataFrame(
                    columns=[
                        "id",
                        "metadata",
                        "content",
                        "embedding",
                        "created_at",
                        "similarity",
                    ]
                )

            logger.debug(f"Converting {len(results)} results to DataFrame")
            df = pd.DataFrame(
                results,
                columns=[
                    "id",
                    "metadata",
                    "content",
                    "embedding",
                    "created_at",
                    "similarity",
                ],
            )
            logger.debug(f"DataFrame shape before metadata expansion: {df.shape}")

            # Retain 'metadata' as a JSON column
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

            with self.conn.cursor() as cur:
                if delete_all:
                    cur.execute(f"TRUNCATE TABLE {settings.VECTOR_STORE_TABLE_NAME}")
                elif ids:
                    # Cast the string IDs to UUID
                    uuid_array = [str(uuid.UUID(id_)) for id_ in ids]
                    cur.execute(
                        f"DELETE FROM {settings.VECTOR_STORE_TABLE_NAME} WHERE id = ANY(%s::uuid[])",
                        (uuid_array,),
                    )
                elif metadata_filter:
                    cur.execute(
                        f"DELETE FROM {settings.VECTOR_STORE_TABLE_NAME} WHERE metadata @> %s",
                        (Json(metadata_filter),),
                    )

            logger.info("Delete operation completed successfully")
        except Exception as e:
            logger.error(f"Error during delete: {e}")
            raise

    def verify_record_exists(self, content: str) -> bool:
        """Verify if a record with given content exists."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"SELECT COUNT(*) FROM {settings.VECTOR_STORE_TABLE_NAME} WHERE content = %s",
                    (content,),
                )
                count = cur.fetchone()[0]
                logger.debug(f"Found {count} records with content: {content}")
                return count > 0
        except Exception as e:
            logger.error(f"Error verifying record: {e}")
            raise
