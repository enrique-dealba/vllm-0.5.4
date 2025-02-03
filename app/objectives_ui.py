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
        "objective_end_time": "2024-05-21T22:30:00.250000+00:00",
        "objective_start_time": "2024-05-21T19:20:00.150000+00:00",
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
            current_value = fields[field_name]

            # Special handling for datetime fields
            if field_name in ["objective_start_time", "objective_end_time"]:
                # Extract just the datetime components without the formatting
                expected_dt = expected_value.replace("T", " ").split("+")[0]
                current_dt = str(current_value).split("tzinfo")[0].strip()
                if expected_dt in current_dt:
                    correct_fields += 1
                continue

            # Handle lists (like sensor_name_list and rso_id_list)
            if isinstance(expected_value, list):
                if sorted(current_value) == sorted(expected_value):
                    correct_fields += 1
                continue

            # Regular field comparison
            if str(current_value) == str(expected_value):
                correct_fields += 1

    accuracy = (correct_fields / total_fields) * 100
    return accuracy, correct_fields, total_fields


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
