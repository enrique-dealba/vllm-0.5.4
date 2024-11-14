from datetime import datetime, timezone

import pandas as pd
from timescale_vector.client import uuid_from_time

from app.backend.vector_store import VectorStore

# Initialize VectorStore
vec_store = VectorStore()

# Read the CSV file
df = pd.read_csv("../data/faq_dataset.csv", sep=";")


# Prepare data for insertion
def prepare_record(row):
    """Prepare a record for insertion into the vector store."""
    content = f"Question: {row['question']}\nAnswer: {row['answer']}"
    embedding = vec_store.get_embedding(content)
    return pd.Series(
        {
            "id": str(uuid_from_time(datetime.utcnow())),
            "metadata": {
                "category": row.get("category", "Uncategorized"),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            "content": content,
            "embedding": embedding,
        }
    )


records_df = df.apply(prepare_record, axis=1)

# Create tables and insert data
vec_store.create_tables()
vec_store.create_index()  # DiskAnnIndex
vec_store.upsert(records_df)

print(f"Inserted {len(records_df)} records into the vector store.")
