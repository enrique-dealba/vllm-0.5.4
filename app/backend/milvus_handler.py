import json
import logging
from pathlib import Path

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections

from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MilvusHandler:
    def __init__(self, collection_name="rag_collection"):
        db_path = settings.MILVUS_DB_PATH
        db_path_obj = Path(db_path).resolve()

        # Ensure the directory exists
        if not db_path_obj.parent.exists():
            db_path_obj.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory for Milvus Lite DB: {db_path_obj.parent}")

        # Ensure the database file exists
        if not db_path_obj.exists():
            db_path_obj.touch()
            logger.info(f"Created Milvus Lite DB file: {db_path_obj}")

        # Construct the URI using as_uri and replace scheme
        # Correct replacement: 'file://' -> 'sqlite:///'
        self.uri = db_path_obj.as_uri().replace("file://", "sqlite:///")
        logger.info(f"Connecting to Milvus Lite with URI: {self.uri}")

        try:
            connections.connect(alias="default", uri=self.uri)
            logger.info("Successfully connected to Milvus Lite.")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus Lite: {e}")
            raise e

        self.collection_name = collection_name

        # Check if collection exists; if not, create it
        if self.collection_name not in self.list_collections():
            self.create_collection()
        self.collection = Collection(name=self.collection_name)

    def list_collections(self):
        return connections.list_collections()

    def create_collection(
        self, embedding_dim=400
    ):  # Adjust based on your embedding model
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=False,
                max_length=36,
            ),
            FieldSchema(
                name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim
            ),
            FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="source_files", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(
                name="json_keys_summary", dtype=DataType.VARCHAR, max_length=5000
            ),
            FieldSchema(
                name="descriptive_labels", dtype=DataType.VARCHAR, max_length=5000
            ),
            FieldSchema(name="context_info", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="num_values", dtype=DataType.INT32),
            FieldSchema(name="priority_level", dtype=DataType.INT32),
        ]
        schema = CollectionSchema(fields=fields, description="RAG system collection")
        Collection(name=self.collection_name, schema=schema)
        logger.info(f"Collection '{self.collection_name}' created successfully.")

    def insert_documents(self, documents: list):
        """Insert a list of documents into the collection.
        Each document is a dict with keys: id, embedding, chunk, source_files, json_keys_summary, descriptive_labels, context_info, num_values, priority_level
        """
        if not documents:
            return

        ids = [doc["id"] for doc in documents]
        embeddings = [doc["embedding"] for doc in documents]
        chunks = [doc["chunk"] for doc in documents]
        source_files = [json.dumps(doc["source_files"]) for doc in documents]
        json_keys_summary = [json.dumps(doc["json_keys_summary"]) for doc in documents]
        descriptive_labels = [
            json.dumps(doc["descriptive_labels"]) for doc in documents
        ]
        context_info = [doc.get("context_info", "") for doc in documents]
        num_values = [doc.get("num_values", 0) for doc in documents]
        priority_level = [doc.get("priority_level", 1) for doc in documents]

        data = [
            ids,
            embeddings,
            chunks,
            source_files,
            json_keys_summary,
            descriptive_labels,
            context_info,
            num_values,
            priority_level,
        ]

        self.collection.insert(data)
        self.collection.flush()

    def query_all_documents(self):
        """Retrieve all documents from the collection."""
        return self.collection.query(
            expr="id != ''",
            output_fields=[
                "id",
                "chunk",
                "source_files",
                "json_keys_summary",
                "descriptive_labels",
                "context_info",
                "num_values",
                "priority_level",
            ],
        )

    def similarity_search(self, query_embedding: list, top_k=5, filter_expr=None):
        """Perform a similarity search with optional filtering."""
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=[
                "chunk",
                "source_files",
                "json_keys_summary",
                "descriptive_labels",
                "context_info",
                "num_values",
                "priority_level",
            ],
        )
        return results

    def disconnect(self):
        connections.disconnect(alias="default")
        logger.info("Disconnected from Milvus Lite.")
