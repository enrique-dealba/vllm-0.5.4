import json
import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from llm_logic import generate_response
from utils import (
    format_summary_collects,
    format_summary_intents,
    parse_collect_requests,
    parse_intents,
)

st.title("Dummy Belief State Summarizer")

# File uploaders
intents_file = st.file_uploader("Upload Intents JSON", type=["json"])
collect_requests_file = st.file_uploader("Upload Collect Requests JSON", type=["json"])

if intents_file and collect_requests_file:
    if st.button("Parse Files"):
        # Parse intents
        intents_data = json.load(intents_file)
        intents_summary = parse_intents(intents_data)
        formatted_summary_intents = format_summary_intents(intents_summary)

        # Parse collect requests
        collect_requests_data = json.load(collect_requests_file)
        collect_requests_summary = parse_collect_requests(collect_requests_data)
        formatted_summary_collect_requests = format_summary_collects(
            collect_requests_summary, "collect_requests"
        )

        # Display parsed data side by side
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Intents")
            st.json(formatted_summary_intents)
        with col2:
            st.subheader("Collect Requests")
            st.json(formatted_summary_collect_requests)

        # Create combined summary string
        combined_summary = f"""
        Intents Summary:
        {json.dumps(formatted_summary_intents, indent=2)}

        Collect Requests Summary:
        {json.dumps(formatted_summary_collect_requests, indent=2)}
        """
        st.session_state["combined_summary"] = combined_summary

st.title("LLM:")

# Input text
user_input = st.text_input("Enter your query:", "")

if st.button("Generate"):
    if not user_input:
        st.warning("Please enter a query.")
    else:
        try:
            # If we have a combined summary, pass it to the generate_response function
            context = st.session_state.get("combined_summary", "")
            query = f"""Given this info: {str(context)}, analyze the info and address the user's request: {user_input}.
            Keep your response to 100 words. Please proceed:"""
            llm_response, execution_time = generate_response(query)
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
