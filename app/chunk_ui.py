# app/chunk_ui.py

import json
import os
import sys
import uuid

import streamlit as st

# Import custom backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.embedding import EmbeddingModel
from backend.milvus_handler import MilvusHandler
from config import settings
from llm_logic import generate_response

st.title("Text File Analysis with Chunk Metadata and Milvus Lite Integration")

# Initialize backend components
embedder = EmbeddingModel()
milvus = MilvusHandler()

# File uploader for .txt file
txt_file = st.file_uploader("Upload a .txt File Containing JSONs", type=["txt"])

if txt_file:
    content = txt_file.read().decode("utf-8")
    chunk_size = settings.CHUNK_SIZE  # 1000 as per config
    chunks = [
        content[i : i + chunk_size] for i in range(0, len(content), chunk_size)
    ]  # Split content into 1000-char chunks

    if st.button("Analyze and Store Chunks"):
        st.subheader("Processing and Storing Chunks")
        inserted_count = 0
        for i, chunk in enumerate(chunks):
            try:
                query = f"""Analyze this text chunk: {chunk}
                Generate structured metadata based on the given schema."""
                llm_response, execution_time = generate_response(query)

                # Generate embeddings for RAG
                # embedding = embedder.encode([chunk])[0]  # Chunk embedding
                embedding = embedder.encode([str(llm_response)])[
                    0
                ]  # Metadata embedding

                # Create a unique ID for the chunk
                chunk_id = str(uuid.uuid4())

                # Prepare document for insertion
                document = {
                    "id": chunk_id,
                    "embedding": embedding,
                    "chunk": chunk,
                    "source_files": llm_response.get("source_files", []),
                    "json_keys_summary": llm_response.get("json_keys_summary", []),
                    "descriptive_labels": llm_response.get("descriptive_labels", {}),
                    "context_info": llm_response.get("context_info", ""),
                    "num_values": llm_response.get("num_values", 0),
                    "priority_level": llm_response.get("priority_level", 1),
                }

                # Insert document into Milvus Lite
                milvus.insert_documents([document])
                inserted_count += 1

                st.success(f"Chunk {i+1} processed and stored successfully.")
            except Exception as e:
                st.error(f"An error occurred while processing chunk {i+1}: {e}")

        st.success(
            f"All chunks have been processed and stored in Milvus Lite. Total inserted: {inserted_count}"
        )

    if st.button("View Stored Chunks"):
        st.subheader("Stored Chunks and Metadata")
        try:
            results = milvus.query_all_documents()

            for result in results:
                # Create a formatted JSON object for each document
                formatted_data = {
                    "ID": result["id"],
                    "Chunk": result["chunk"],
                    "Source Files": json.loads(result["source_files"]),
                    "JSON Keys Summary": json.loads(result["json_keys_summary"]),
                    "Descriptive Labels": json.loads(result["descriptive_labels"]),
                    "Context Info": result.get("context_info", "N/A"),
                    "Number of Values": result.get("num_values", 0),
                    "Priority Level": result.get("priority_level", 1),
                }

                # Display as collapsible JSON
                with st.expander(f"Document ID: {result['id']}"):
                    st.json(formatted_data)
        except Exception as e:
            st.error(f"An error occurred while retrieving data: {e}")

    st.markdown("---")

    st.subheader("Perform Similarity Search")
    query = st.text_input("Enter your query for similarity search:")
    if st.button("Search"):
        if query:
            try:
                # Generate embedding for the query
                query_embedding = embedder.encode([query])[0]
                # Perform similarity search
                results = milvus.similarity_search(query_embedding, top_k=5)

                st.subheader("Search Results")

                for hit in results[0]:
                    # Create a formatted JSON object for each search result
                    result_data = {
                        "Score": hit.score,
                        "Chunk": hit.entity.get("chunk"),
                        "Source Files": json.loads(hit.entity.get("source_files")),
                        "JSON Keys Summary": json.loads(
                            hit.entity.get("json_keys_summary")
                        ),
                        "Descriptive Labels": json.loads(
                            hit.entity.get("descriptive_labels")
                        ),
                        "Context Info": hit.entity.get("context_info"),
                        "Number of Values": hit.entity.get("num_values"),
                        "Priority Level": hit.entity.get("priority_level"),
                    }

                    # Display as collapsible JSON with score in the header
                    with st.expander(f"Result (Score: {hit.score:.4f})"):
                        st.json(result_data)

            except Exception as e:
                st.error(f"An error occurred during the search: {e}")
        else:
            st.warning("Please enter a query to search.")

    st.markdown("---")

    st.subheader("Advanced: Filtered Similarity Search")
    filter_field = st.selectbox(
        "Select Filter Field", ["source_files", "created_at", "priority_level"]
    )
    filter_value = st.text_input("Enter Filter Value for Filtering:")

    if st.button("Filtered Search"):
        if query and filter_field and filter_value:
            try:
                # Generate embedding for the query
                query_embedding = embedder.encode([query])[0]

                # Create filter expression
                if filter_field in [
                    "source_files",
                    "json_keys_summary",
                    "descriptive_labels",
                ]:
                    # For list or dict fields stored as JSON strings
                    expr = f"{filter_field} CONTAINS '{filter_value}'"
                else:
                    expr = f"{filter_field} == {filter_value}"

                # Perform similarity search with filter
                results = milvus.similarity_search(
                    query_embedding, top_k=5, filter_expr=expr
                )

                st.subheader("Filtered Search Results")

                # Display filter information
                st.info(f"Applied Filter: {filter_field} = {filter_value}")

                for hit in results[0]:
                    # Create a formatted JSON object for each filtered result
                    result_data = {
                        "Score": hit.score,
                        "Chunk": hit.entity.get("chunk"),
                        "Source Files": json.loads(hit.entity.get("source_files")),
                        "JSON Keys Summary": json.loads(
                            hit.entity.get("json_keys_summary")
                        ),
                        "Descriptive Labels": json.loads(
                            hit.entity.get("descriptive_labels")
                        ),
                        "Context Info": hit.entity.get("context_info"),
                        "Number of Values": hit.entity.get("num_values"),
                        "Priority Level": hit.entity.get("priority_level"),
                    }

                    # Display as collapsible JSON with score in the header
                    with st.expander(f"Filtered Result (Score: {hit.score:.4f})"):
                        st.json(result_data)

            except Exception as e:
                st.error(f"An error occurred during the filtered search: {e}")
        else:
            st.warning("Please enter both query and filter criteria.")

    st.markdown("---")

    st.subheader("Download Milvus DB")
    db_file_path = settings.MILVUS_DB_PATH
    if os.path.exists(db_file_path):
        with open(db_file_path, "rb") as db_file:
            db_bytes = db_file.read()
            st.download_button(
                label="Download Milvus DB",
                data=db_bytes,
                file_name="milvus_demo.db",
                mime="application/octet-stream",
            )
    else:
        st.warning("Milvus DB file not found.")

    st.markdown("---")

    if st.button("Disconnect Milvus"):
        milvus.disconnect()
        st.success("Disconnected from Milvus Lite.")
