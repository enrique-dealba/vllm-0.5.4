import logging
import os
import sys
import uuid
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

# Import custom backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_logic import generate_response

from app.backend.vector_store import VectorStore
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


st.title("Text File Analysis with Chunk Metadata and PostgreSQL/Timescale Integration")

# Initialize backend components
try:
    vec_store = VectorStore()
    st.success("Connected to vector store successfully")
except Exception as e:
    st.error(f"Failed to connect to vector store: {e}")
    st.stop()

# File uploader for .txt file
txt_file = st.file_uploader("Upload a .txt File Containing JSONs", type=["txt"])

if txt_file:
    content = txt_file.read().decode("utf-8")
    chunk_size = settings.CHUNK_SIZE
    chunks = [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]

    if st.button("Analyze and Store Chunks"):
        st.subheader("Processing and Storing Chunks")
        inserted_count = 0
        metadata_records = []

        # Create tables and index if they don't exist
        with st.spinner("Setting up database tables and indexes..."):
            try:
                vec_store.create_tables()
                vec_store.create_index()
                st.success("Database setup complete")
            except Exception as e:
                st.error(f"Database setup failed: {e}")
                st.stop()

        progress_bar = st.progress(0)
        for i, chunk in enumerate(chunks):
            try:
                query = f"""Analyze this text chunk: {chunk}
                Generate structured metadata based on the given schema."""
                llm_response, execution_time = generate_response(query)

                # chunk_embedding = vec_store.get_embedding(chunk)  # TODO: Check how we can use this
                metadata_embedding = vec_store.get_embedding(str(llm_response))
                chunk_id = str(uuid.uuid4())

                document = {
                    "id": chunk_id,
                    "metadata": llm_response.model_dump(),
                    "content": chunk,
                    "embedding": metadata_embedding,  # Using metadata_embedding as main embedding
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }

                metadata_records.append(document)
                inserted_count += 1
                progress_bar.progress((i + 1) / len(chunks))

            except Exception as e:
                st.error(f"Error processing chunk {i+1}: {e}")
                logger.error(f"Chunk processing error: {e}", exc_info=True)

        if metadata_records:
            try:
                records_df = pd.DataFrame(metadata_records)
                vec_store.upsert(records_df)
                st.success(f"Successfully stored {inserted_count} chunks")
            except Exception as e:
                st.error(f"Failed to store chunks: {e}")
                logger.error("Chunk storage error", exc_info=True)

    if st.button("View Stored Chunks"):
        st.subheader("Stored Chunks and Metadata")
        try:
            with st.spinner("Fetching stored chunks..."):
                results = vec_store.search("", limit=100)
                if results.empty:
                    st.info("No chunks found in the database")
                else:
                    for _, row in results.iterrows():
                        formatted_data = {
                            "ID": row["id"],
                            "Chunk": row["content"],
                            "Source Files": row["metadata"].get("source_files", []),
                            "JSON Keys": row["metadata"].get("json_keys_summary", []),
                            "Descriptive Labels": row["metadata"].get(
                                "descriptive_labels", {}
                            ),
                            "Context": row["metadata"].get("context_info", ""),
                            "Number of Values": row["metadata"].get("num_values", 0),
                            "Priority Level": row["metadata"].get("priority_level", 1),
                            "Created At": row["created_at"],
                        }

                        with st.expander(
                            f"Document ID: {formatted_data['ID']} (Priority: {formatted_data['Priority Level']})"
                        ):
                            st.json(formatted_data)
        except Exception as e:
            st.error(f"Failed to retrieve chunks: {e}")
            logger.error("Chunk retrieval error", exc_info=True)

    st.markdown("---")

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Enter your query for similarity search:")
    with col2:
        limit = st.number_input(
            "Number of results", min_value=1, max_value=100, value=5
        )

    if st.button("Search"):
        if query:
            try:
                with st.spinner("Searching..."):
                    results = vec_store.search(query, limit=limit)

                if results.empty:
                    st.info("No matching results found")
                else:
                    st.subheader("Search Results")
                    for _, row in results.iterrows():
                        result_data = {
                            "ID": row["id"],
                            "Chunk": row["content"],
                            "Source Files": row["metadata"].get("source_files", []),
                            "JSON Keys": row["metadata"].get("json_keys_summary", []),
                            "Descriptive Labels": row["metadata"].get(
                                "descriptive_labels", {}
                            ),
                            "Context": row["metadata"].get("context_info", ""),
                            "Number of Values": row["metadata"].get("num_values", 0),
                            "Priority Level": row["metadata"].get("priority_level", 1),
                            "Created At": row["created_at"],
                            "Distance": row["distance"],
                        }

                        with st.expander(
                            f"Result (Distance: {result_data['Distance']:.4f}, Priority: {result_data['Priority Level']})"
                        ):
                            st.json(result_data)
            except Exception as e:
                st.error(f"Search failed: {e}")
                logger.error("Search error", exc_info=True)
        else:
            st.warning("Please enter a query to search")

    st.markdown("---")
