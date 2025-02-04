import os
import sys

import streamlit as st

# Ensure the app directory is on the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.langchain_structured_outputs import generate_objective_response
from app.utils import calculate_field_accuracy, get_displayable_fields

# Set up the Streamlit interface
st.title("LLM")

# Create the input field
user_input = st.text_input("Enter spaceplan objective:", "")

# Handle the generate button
if st.button("Generate Schema"):
    if not user_input:
        st.warning("Please enter a query.")
    else:
        try:
            assert (
                settings.USE_STRUCTURED_OUTPUT
            ), "Structured output is disabled in settings."

            response, time_details = generate_objective_response(user_input)
            st.info(f"Raw Response: {response}")
            fields = get_displayable_fields(response)

            # Display the response
            st.json(fields)
            st.info(f"Total Execution Time: {time_details:.2f} seconds")
            # Field accuracy
            accuracy = calculate_field_accuracy(fields)
            accuracy, correct_fields, total_fields = calculate_field_accuracy(fields)
            st.info(
                f"Percent Correct Fields: {accuracy:.1f}% ({correct_fields}/{total_fields} fields)"
            )

        except Exception as e:
            st.error(f"An error occurred: {e}")
