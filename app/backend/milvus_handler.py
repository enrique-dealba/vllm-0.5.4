import json

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections

from ..config import settings


class MilvusHandler:
    def __init__(self, collection_name="rag_collection"):
        self.uri = f"sqlite:///{settings.MILVUS_DB_PATH}"
        connections.connect(alias="default", uri=self.uri)
        self.collection_name = collection_name
        if self.collection_name not in self.list_collections():
            self.create_collection()
        self.collection = Collection(name=self.collection_name)

    def list_collections(self):
        return connections.list_collections()

    def create_collection(
        self, embedding_dim=settings.EMBEDDING_DIM
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
        print(f"Collection '{self.collection_name}' created successfully.")

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
