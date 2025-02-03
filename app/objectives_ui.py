import os
import sys

import streamlit as st

# Ensure the app directory is on the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from llm_logic import generate_response
from utils import get_displayable_fields

st.title("LLM")

user_input = st.text_input("Enter your objective:", "")

if st.button("Generate Schema"):
    if not user_input:
        st.warning("Please enter a query.")
    else:
        try:
            assert (
                settings.USE_STRUCTURED_OUTPUT
            ), "Structured output is disabled in settings."

            # 1. Generate the initial response using the base objective schema
            initial_response, time_initial = generate_response(user_input)
            fields_initial = get_displayable_fields(initial_response)

            # 2. Determine which objective type was predicted
            objective_type = fields_initial.get("objective_name", None)

            # List the objective types that need detailed processing
            detailed_objective_types = [
                "CatalogMaintenanceObjective",
                "PeriodicRevisitObjective",
                "SearchObjective",
                "DataEnrichmentObjective",
                "SpectralClearingObjective",
            ]

            # 3. If the initial response indicates one of these objective types,
            # re-run the chain with a more detailed schema.
            if objective_type in detailed_objective_types:
                # Back up the current schema configuration
                original_schema = settings.LLM_RESPONSE_SCHEMA
                # For catalog maintenance objectives we want a template version;
                # for the others, we assume the detailed schema has the same name.
                if objective_type == "CatalogMaintenanceObjective":
                    settings.LLM_RESPONSE_SCHEMA = "CatalogMaintenanceObjective"
                else:
                    settings.LLM_RESPONSE_SCHEMA = objective_type

                detailed_response, time_detailed = generate_response(user_input)

                # Restore the original schema setting after obtaining the detailed response
                # settings.LLM_RESPONSE_SCHEMA = original_schema

                fields = get_displayable_fields(detailed_response)
                total_time = time_initial + time_detailed
            else:
                # If no detailed chain is needed, use the initial response.
                fields = fields_initial
                total_time = time_initial

            # Display the JSON fields nicely and show the total execution time.
            st.json(fields)
            st.info(f"Total Execution Time: {total_time:.2f} seconds")
        except Exception as e:
            st.error(f"An error occurred: {e}")
