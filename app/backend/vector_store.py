import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from timescale_vector import client

from app.backend.embedding import EmbeddingModel
from app.backend.ts_config import DISKANN_INDEX_PARAMS, TIME_PARTITION_INTERVAL
from app.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self):
        """Initialize VectorStore with Timescale and embedding model."""
        self.embedder = EmbeddingModel()
        try:
            self.vec_client = client.Sync(
                service_url=settings.TIMESCALE_SERVICE_URL,
                table_name=settings.VECTOR_STORE_TABLE_NAME,
                embedding_dimensions=settings.VECTOR_STORE_EMBEDDING_DIMENSIONS,
                time_partition_interval=TIME_PARTITION_INTERVAL,
                distance_type="cosine",  # Explicitly set distance type
            )
            logger.info("Connected to Timescale Vector store successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to Timescale Vector store: {e}")
            raise e

    def create_tables(self) -> None:
        """Create necessary tables in the database."""
        try:
            self.vec_client.create_tables()
            logger.info(f"Tables created in '{settings.VECTOR_STORE_TABLE_NAME}'.")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise e

    def create_index(self) -> None:
        """Create the DiskANN index (recommended for most use cases)."""
        try:
            self.vec_client.create_embedding_index(
                client.DiskAnnIndex(**DISKANN_INDEX_PARAMS)
            )
            logger.info("DiskANN index created successfully.")
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise e

    def upsert(self, df: pd.DataFrame) -> None:
        """Insert or update records with time-based UUIDs."""
        try:
            # Convert records to list of tuples with time-based UUIDs
            records = []
            for _, row in df.iterrows():
                timestamp = datetime.now()
                if "created_at" in row["metadata"]:
                    timestamp = datetime.fromisoformat(
                        row["metadata"]["created_at"].replace("Z", "+00:00")
                    )

                record_id = client.uuid_from_time(timestamp)
                records.append(
                    (record_id, row["metadata"], row["content"], row["embedding"])
                )

            self.vec_client.upsert(records)
            logger.info(f"Inserted {len(df)} records.")
        except Exception as e:
            logger.error(f"Error during upsert: {e}")
            raise e

    def search(
        self,
        query_text: str,
        limit: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        return_dataframe: bool = True,
    ) -> Union[List[Tuple[Any, ...]], pd.DataFrame]:
        """Perform similarity search with optional time filtering."""
        try:
            query_embedding = self.get_embedding(query_text)
            search_args = {"limit": limit}

            # Add metadata filter if provided
            if metadata_filter:
                search_args["filter"] = metadata_filter

            # Add time range filter if provided
            if time_range:
                start_date, end_date = time_range
                search_args["uuid_time_filter"] = client.UUIDTimeRange(
                    start_date, end_date
                )

            # Add query parameters for DiskANN
            search_args["query_params"] = client.DiskAnnIndexParams(
                rescore=50,  # Default rescore value
                search_list_size=100,  # Default search list size
            )

            results = self.vec_client.search(query_embedding, **search_args)

            if return_dataframe:
                return self._create_dataframe_from_results(results)
            return results

        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise e

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        text = text.replace("\n", " ")
        try:
            embedding = self.embedder.encode([text])[0]
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise e

    def _create_dataframe_from_results(self, results) -> pd.DataFrame:
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
            raise e

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
            raise e
