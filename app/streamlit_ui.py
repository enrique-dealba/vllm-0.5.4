import os
import sys
from typing import Any

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from llm_logic import generate_response

FIELD_DISPLAY_CONFIG = {
    "response": {"title": "Response", "display_format": lambda x: x, "is_list": False},
    "sources": {
        "title": "Sources",
        "display_format": lambda x: f"- {x}",
        "is_list": True,
    },
    "evidence": {
        "title": "Supporting Evidence",
        "display_format": lambda x: f"- {x}",
        "is_list": True,
    },
    "confidence": {
        "title": "Confidence",
        "display_format": lambda x: f"{round(float(x) * 100, 1)}%",
        "is_list": False,
    },
    "classification_marking": {
        "title": "Classification",
        "display_format": lambda x: x,
        "is_list": False,
    },
    "data_mode": {
        "title": "Data Mode",
        "display_format": lambda x: x,
        "is_list": False,
    },
    "collect_request_type": {
        "title": "Collection Request Type",
        "display_format": lambda x: x,
        "is_list": False,
    },
    "orbital_regime": {
        "title": "Orbital Regime",
        "display_format": lambda x: x,
        "is_list": False,
    },
    "patience_minutes": {
        "title": "Patience Time (Minutes)",
        "display_format": lambda x: str(x),
        "is_list": False,
    },
    "end_time_offset_minutes": {
        "title": "End Time Offset (Minutes)",
        "display_format": lambda x: str(x),
        "is_list": False,
    },
    "priority": {
        "title": "Priority Level",
        "display_format": lambda x: str(x),
        "is_list": False,
    },
    "key_points": {
        "title": "Key Points",
        "display_format": lambda x: f"- {x}",
        "is_list": True,
    },
    "summary": {"title": "Summary", "display_format": lambda x: x, "is_list": False},
    "categories": {
        "title": "Categories",
        "display_format": lambda x: f"- {x}",
        "is_list": True,
    },
    "priority_level": {
        "title": "Priority Level",
        "display_format": lambda x: str(x),
        "is_list": False,
    },
}


def display_field(field_name: str, value: Any) -> None:
    """Display a field based on its configuration."""
    if not value:  # Skip empty values
        return

    config = FIELD_DISPLAY_CONFIG.get(
        field_name,
        {
            "title": field_name.replace("_", " ").title(),
            "display_format": lambda x: x,
            "is_list": isinstance(value, (list, tuple)),
        },
    )

    st.subheader(config["title"])

    if config["is_list"]:
        for item in value:
            st.write(config["display_format"](item))
    else:
        st.write(config["display_format"](value))


def display_response(llm_response: Any) -> None:
    """Display all available fields from the LLM response."""
    # Always display response field first if it exists
    if hasattr(llm_response, "response"):
        display_field("response", llm_response.response)

    # Display all other fields
    for field_name in dir(llm_response):
        # Skip private attributes and already displayed response
        if not field_name.startswith("_") and field_name != "response":
            value = getattr(llm_response, field_name)
            if not callable(value):  # Skip methods
                display_field(field_name, value)


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
                display_response(llm_response)
            else:
                st.write(llm_response)

        except Exception as e:
            st.error(f"An error occurred: {e}")
