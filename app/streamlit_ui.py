import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from llm_logic import generate_response

st.title("LLM")

# Input text
user_input = st.text_input("Enter your query:", "")

if st.button("Generate"):
    if not user_input:
        st.warning("Please enter a query.")
    else:
        try:
            llm_response, execution_time = generate_response(user_input)

            st.subheader("Response:")
            if settings.USE_STRUCTURED_OUTPUT:
                st.write(llm_response.response)

                if hasattr(llm_response, "sources") and llm_response.sources:
                    st.subheader("Sources:")
                    for source in llm_response.sources:
                        st.write(f"- {source}")

                if hasattr(llm_response, "confidence"):
                    st.subheader("Confidence:")
                    st.write(f"{llm_response.confidence:.2f}")
            else:
                st.write(llm_response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
