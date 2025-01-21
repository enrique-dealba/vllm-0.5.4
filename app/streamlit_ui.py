import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from llm_logic import generate_response
from utils import display_response

st.title("LLM")

# Input text
user_input = st.text_input("Enter your query:", "")

if st.button("Generate"):
    if not user_input:
        st.warning("Please enter a query.")
    else:
        try:
            llm_response, execution_time = generate_response(user_input)

            if settings.USE_STRUCTURED_OUTPUT:
                display_response(llm_response, writer_func=st.write)
            else:
                st.write(llm_response)

        except Exception as e:
            st.error(f"An error occurred: {e}")
