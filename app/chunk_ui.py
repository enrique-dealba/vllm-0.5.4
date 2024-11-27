import logging
import os
import sys
import uuid
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_logic import generate_response

from app.backend.rag import RAG
from app.backend.vector_store import VectorStore
from app.config import settings
from app.schemas.llm_responses import ChunkMetadata
from app.utils import handle_chunk_processing_errors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_chunk(chunk: str, vec_store: VectorStore) -> dict:
    @handle_chunk_processing_errors
    def _process_single_chunk():
        query = f"""Analyze this text chunk: {chunk}
        Generate structured metadata based on the given schema."""

        try:
            llm_response, _ = generate_response(query)
        except Exception as e:
            logger.warning(f"LLM response generation failed: {e}")
            # Create fallback metadata
            llm_response = ChunkMetadata(
                # source_files=[],
                categories=["Other"],
                summary=f"Failed to process chunk: {str(e)}",
                priority_level=1,
            )

        try:
            metadata_embedding = vec_store.get_embedding(str(llm_response.model_dump()))
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return None

        return {
            "id": str(uuid.uuid4()),
            "metadata": llm_response.model_dump(),
            "content": chunk,
            "embedding": metadata_embedding,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    return _process_single_chunk()


st.title("PostgreSQL RAG")

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
        processed_chunks = []
        failed_chunks = []

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
            result = process_chunk(chunk, vec_store)
            if result:
                processed_chunks.append(result)
            else:
                failed_chunks.append(i)
            progress_bar.progress((i + 1) / len(chunks))

        if processed_chunks:
            try:
                records_df = pd.DataFrame(processed_chunks)
                vec_store.upsert(records_df)
                st.success(f"Successfully stored {len(processed_chunks)} chunks")
            except Exception as e:
                st.error(f"Failed to store chunks: {e}")
                logger.error("Chunk storage error", exc_info=True)

        if failed_chunks:
            st.warning(
                f"Failed to process {len(failed_chunks)} chunks at indices: {failed_chunks}"
            )

    if st.button("View Stored Chunks"):
        st.subheader("Stored Chunks and Metadata")
        try:
            logger.info("View Stored Chunks button clicked.")
            with st.spinner("Fetching stored chunks..."):
                logger.info("Attempting to fetch stored chunks.")
                results = vec_store.search("", limit=100)  # Retrieves all records
                logger.debug(f"Number of chunks fetched: {len(results)}")

                if results.empty:
                    logger.info("No chunks found in the database.")
                    st.info("No chunks found in the database")
                else:
                    logger.info(f"{len(results)} chunks retrieved successfully.")
                    st.write(f"Total Chunks Retrieved: {len(results)}")

                    for index, row in results.iterrows():
                        logger.debug(f"Processing chunk at index {index}.")
                        metadata = row.get("metadata", {})
                        created_at = row.get("created_at", "N/A")
                        if isinstance(created_at, pd.Timestamp):
                            created_at = created_at.strftime("%Y-%m-%d %H:%M:%S %Z")

                        formatted_data = {
                            "ID": row.get("id", "N/A"),
                            "Chunk": row.get("content", ""),
                            # "Source Files": metadata.get("source_files", []),
                            "Categories": metadata.get("categories", []),
                            "Summary": metadata.get("summary", ""),
                            "Key Points": metadata.get("key_points", []),
                            # "Context": metadata.get("context_info", ""),
                            "Priority Level": metadata.get("priority_level", 1),
                            "Created At": created_at,
                        }

                        logger.info(
                            f"Formatted data for chunk ID {formatted_data['ID']}: {formatted_data}"
                        )

                        with st.expander(
                            f"Document ID: {formatted_data['ID']} (Priority: {formatted_data['Priority Level']})"
                        ):
                            st.json(formatted_data)
        except Exception as e:
            logger.error("Chunk retrieval error", exc_info=True)
            st.error(f"Failed to retrieve chunks: {e}")
            logger.info("Displayed error message to the user.")

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
                            # "Source Files": row["metadata"].get("source_files", []),
                            "Categories": row["metadata"].get("categories", []),
                            "Summary": row["metadata"].get("summary", ""),
                            "Key Points": row["metadata"].get("key_points", []),
                            # "Context": row["metadata"].get("context_info", ""),
                            "Priority Level": row["metadata"].get("priority_level", 1),
                            "Created At": row["created_at"],
                            "Distance": row.get("similarity", 0),
                        }

                        with st.expander(
                            f"Result (Similarity: {result_data['Distance']:.4f}, Priority: {result_data['Priority Level']})"
                        ):
                            st.json(result_data)

                    # Add RAG-based answer generation using existing results
                    st.subheader("Response")
                    with st.spinner("Generating answer based on retrieved context..."):
                        try:
                            rag = RAG()
                            # Pass the existing results to avoid duplicate search
                            answer = rag.generate_answer(
                                query, existing_context=results
                            )

                            st.write(answer["response"])

                            st.write("### Details")
                            st.write(
                                f"Execution Time: {answer['execution_time_seconds']:.2f} seconds"
                            )

                        except Exception as e:
                            st.error(f"Failed to generate answer: {e}")
                            logger.error("Answer generation error", exc_info=True)

            except Exception as e:
                st.error(f"Search failed: {e}")
                logger.error("Search error", exc_info=True)
        else:
            st.warning("Please enter a query to search")

    st.markdown("---")
