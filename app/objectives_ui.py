import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from llm_logic import generate_objective
from utils import get_displayable_fields

st.title("LLM")

user_input = st.text_input("Enter your objective:", "")

if st.button("Generate Schema"):
    if not user_input:
        st.warning("Please enter a query.")
    else:
        try:
            assert settings.USE_STRUCTURED_OUTPUT
            llm_response, execution_time = generate_objective(user_input)
            fields = get_displayable_fields(llm_response)

            # Display as JSON with formatting
            st.json(fields)

            # Optionally, you could also display execution time
            st.info(f"Execution time: {execution_time:.2f} seconds")
        except Exception as e:
            st.error(f"An error occurred: {e}")
