import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_logic import generate_response

st.title("LLM")

# Input text
user_input = st.text_input("Enter your query:", "")

if st.button("Generate"):
    if not user_input:
        st.warning("Please enter a query.")
    else:
        try:
            generated_text, execution_time = generate_response(user_input)

            if generated_text:
                st.subheader("Response:")
                st.write(generated_text)
                st.write(f"Execution Time: {execution_time:.4f} seconds.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
