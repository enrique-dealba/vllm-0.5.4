import logging

import streamlit as st

from app.backend.rag import RAG
from app.backend.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_components():
    try:
        vec_store = VectorStore()
        st.success("Connected to vector store successfully")
        return vec_store
    except Exception as e:
        st.error(f"Failed to connect to vector store: {e}")
        st.stop()


def main():
    st.title("PostgreSQL RAG")

    # Initialize components
    vec_store = initialize_components()

    # Search interface
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Enter your search query:")
    with col2:
        limit = st.number_input(
            "Number of results", min_value=1, max_value=100, value=3
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

                    # Display search results
                    for _, row in results.iterrows():
                        result_data = {
                            "ID": row["id"],
                            "Chunk": row["content"],
                            "Categories": row["metadata"].get("categories", []),
                            "Summary": row["metadata"].get("summary", ""),
                            "Key Points": row["metadata"].get("key_points", []),
                            "Priority Level": row["metadata"].get("priority_level", 1),
                            "Created At": row["created_at"],
                            "Distance": row.get("similarity", 0),
                        }

                        with st.expander(
                            f"Result (Similarity: {result_data['Distance']:.4f}, Priority: {result_data['Priority Level']})"
                        ):
                            st.json(result_data)

                    # Generate RAG-based answer
                    st.subheader("Generated Response")
                    with st.spinner("Generating answer based on retrieved context..."):
                        try:
                            rag = RAG()
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


if __name__ == "__main__":
    main()
