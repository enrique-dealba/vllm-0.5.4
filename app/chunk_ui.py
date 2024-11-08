import os
import sys

import streamlit as st

# Import custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from llm_logic import generate_response

st.title("Text File Analysis with Chunk Metadata")

# File uploader for .txt file
txt_file = st.file_uploader("Upload a .txt File Containing JSONs", type=["txt"])

if txt_file:
    content = txt_file.read().decode("utf-8")
    chunk_size = 1000  # prev: 1000,
    chunks = [
        content[i : i + chunk_size] for i in range(0, len(content), chunk_size)
    ]  # Split content into 100-char chunks

    if st.button("Analyze Chunks"):
        st.subheader("Chunk Analysis")
        chunk_metadata_dict = {}

        for i, chunk in enumerate(chunks):
            try:
                query = f"""Analyze this text chunk: {chunk}
                Generate structured metadata based on the given schema."""
                llm_response, execution_time = generate_response(query)

                # Map chunk to its corresponding LLM response
                chunk_metadata_dict[chunk] = dict(llm_response)
            except Exception as e:
                st.error(f"An error occurred while processing chunk {i+1}: {e}")

        # Display all chunks and their metadata after the loop
        for chunk, metadata in chunk_metadata_dict.items():
            st.write("**Chunk:**")
            st.text(chunk)
            st.write("**Metadata:**")

            if settings.USE_STRUCTURED_OUTPUT:
                if "source_files" in metadata and metadata["source_files"]:
                    st.subheader("Source Files:")
                    for source in metadata["source_files"]:
                        st.write(f"- {source}")
                if "json_keys_summary" in metadata and metadata["json_keys_summary"]:
                    st.subheader("JSON Keys Summary:")
                    for key in metadata["json_keys_summary"]:
                        st.write(f"- {key}")
                if "descriptive_labels" in metadata and metadata["descriptive_labels"]:
                    st.subheader("Descriptive Labels:")
                    for key, label in metadata["descriptive_labels"].items():
                        st.write(f"- {key}: {label}")
                if "context_info" in metadata and metadata["context_info"]:
                    st.subheader("Context Info:")
                    st.write(metadata["context_info"])
                if "num_values" in metadata and metadata["num_values"]:
                    st.subheader("Number of Values:")
                    st.write(metadata["num_values"])
                if "priority_level" in metadata:
                    st.subheader("Priority Level:")
                    st.write(metadata["priority_level"])
            else:
                st.write(metadata)

        # Store the dictionary for future reference
        st.session_state["chunk_metadata_dict"] = chunk_metadata_dict
