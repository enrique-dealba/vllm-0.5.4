import os
import sys

import streamlit as st

# Ensure the app directory is on the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.langchain_structured_outputs import generate_objective_response
from app.utils import get_displayable_fields


def calculate_field_accuracy(fields):
    expected_fields = {
        "classification_marking": "U",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "data_mode": "TEST",
        "end_time_offset_minutes": 25,
        "objective_end_time": "datetime.datetime(2024, 5, 21, 22, 30, 0, 250000, tzinfo=TzInfo(UTC))",
        "objective_start_time": "datetime.datetime(2024, 5, 21, 19, 20, 0, 150000, tzinfo=TzInfo(UTC))",
        "orbital_regime": "LEO",
        "patience_minutes": 10,
        "priority": 12,
        "rso_id_list": [],
        "sensor_name_list": ["RME02", "LMNT01"],
    }

    correct_fields = 0
    total_fields = len(expected_fields)

    for field_name, expected_value in expected_fields.items():
        if field_name in fields:
            if str(fields[field_name]) == str(expected_value):
                correct_fields += 1

    accuracy = (correct_fields / total_fields) * 100
    return accuracy


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
            fields = get_displayable_fields(response)

            # Display the response
            st.json(fields)
            st.info(f"Total Execution Time: {time_details:.2f} seconds")
            # Field accuracy
            accuracy = calculate_field_accuracy(fields)
            st.info(f"Percent Correct Fields: {accuracy:.1f}%")

        except Exception as e:
            st.error(f"An error occurred: {e}")
