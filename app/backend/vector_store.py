import logging
import time
from datetime import datetime
from typing import Any, List, Optional, Tuple, Union

import pandas as pd
from timescale_vector import client

from app.backend.embedding import EmbeddingModel
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
                time_partition_interval=settings.VECTOR_STORE_TIME_PARTITION_INTERVAL,
            )
            logger.info("Connected to Timescale Vector store successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to Timescale Vector store: {e}")
            raise e

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding using sentence-transformers."""
        text = text.replace("\n", " ")
        start_time = time.time()
        embedding = self.embedder.encode([text])[0]
        elapsed_time = time.time() - start_time
        logger.info(f"Embedding generated in {elapsed_time:.3f} seconds")
        return embedding

    def create_tables(self) -> None:
        """Create necessary tables in the database."""
        try:
            self.vec_client.create_tables()
            logger.info(
                f"Tables created in Timescale Vector store '{settings.VECTOR_STORE_TABLE_NAME}'."
            )
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise e

    def create_index(self) -> None:
        """Create the StreamingDiskANN index."""
        try:
            self.vec_client.create_embedding_index(client.DiskAnnIndex())
            logger.info("StreamingDiskANN index created successfully.")
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise e

    def drop_index(self) -> None:
        """Drop the StreamingDiskANN index."""
        try:
            self.vec_client.drop_embedding_index()
            logger.info("StreamingDiskANN index dropped successfully.")
        except Exception as e:
            logger.error(f"Error dropping index: {e}")
            raise e

    def upsert(self, df: pd.DataFrame) -> None:
        """Insert or update records in the database."""
        try:
            records = df.to_records(index=False)
            self.vec_client.upsert(list(records))
            logger.info(
                f"Inserted {len(df)} records into '{settings.VECTOR_STORE_TABLE_NAME}'."
            )
        except Exception as e:
            logger.error(f"Error during upsert operation: {e}")
            raise e

    def search(
        self,
        query_text: str,
        limit: int = 5,
        metadata_filter: Union[dict, List[dict]] = None,
        predicates: Optional[client.Predicates] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        return_dataframe: bool = True,
    ) -> Union[List[Tuple[Any, ...]], pd.DataFrame]:
        """Perform similarity search."""
        try:
            query_embedding = self.get_embedding(query_text)

            start_time = time.time()

            search_args = {"limit": limit}

            if metadata_filter:
                search_args["filter"] = metadata_filter

            if predicates:
                search_args["predicates"] = predicates

            if time_range:
                start_date, end_date = time_range
                search_args["uuid_time_filter"] = client.UUIDTimeRange(
                    start_date, end_date
                )

            results = self.vec_client.search(query_embedding, **search_args)
            elapsed_time = time.time() - start_time

            logger.info(f"Vector search completed in {elapsed_time:.3f} seconds")

            if return_dataframe:
                return self._create_dataframe_from_results(results)
            else:
                return results
        except Exception as e:
            logger.error(f"Error during search operation: {e}")
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
            df["id"] = df["id"].astype(str)
            return df
        except Exception as e:
            logger.error(f"Error formatting search results: {e}")
            raise e

    def delete(
        self,
        ids: List[str] = None,
        metadata_filter: dict = None,
        delete_all: bool = False,
    ) -> None:
        """Delete records from the vector database."""
        try:
            if sum(bool(x) for x in (ids, metadata_filter, delete_all)) != 1:
                raise ValueError(
                    "Provide exactly one of: ids, metadata_filter, or delete_all"
                )

            if delete_all:
                self.vec_client.delete_all()
                logger.info(
                    f"Deleted all records from '{settings.VECTOR_STORE_TABLE_NAME}'."
                )
            elif ids:
                self.vec_client.delete_by_ids(ids)
                logger.info(
                    f"Deleted {len(ids)} records from '{settings.VECTOR_STORE_TABLE_NAME}'."
                )
            elif metadata_filter:
                self.vec_client.delete_by_metadata(metadata_filter)
                logger.info(
                    f"Deleted records matching metadata filter from '{settings.VECTOR_STORE_TABLE_NAME}'."
                )
        except Exception as e:
            logger.error(f"Error during delete operation: {e}")
            raise e
