import asyncio
import json
import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from stream_llm_logic import generate_stream_response
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

            # Generate streamed response
            response_gen = generate_stream_response(query)

            st.subheader("Response:")
            if settings.USE_STRUCTURED_OUTPUT:
                # Initialize placeholders for structured data
                response_container = st.empty()
                sources_container = st.empty()
                confidence_container = st.empty()

                async def stream_structured_response():
                    accumulated_response = {}
                    async for chunk in response_gen:
                        accumulated_response.update(chunk)
                        response_container.write(
                            accumulated_response.get("response", "")
                        )
                        if "sources" in accumulated_response:
                            sources_container.write(
                                [
                                    f"- {source}"
                                    for source in accumulated_response["sources"]
                                ]
                            )
                        if "confidence" in accumulated_response:
                            confidence_container.write(
                                f"{accumulated_response['confidence']:.2f}"
                            )
                        await asyncio.sleep(0.1)  # Small delay for streaming effect

                asyncio.run(stream_structured_response())
            else:
                # For unstructured output
                response_placeholder = st.empty()

                async def stream_unstructured_response():
                    accumulated_text = ""
                    async for chunk in response_gen:
                        accumulated_text += chunk
                        response_placeholder.write(accumulated_text)
                        await asyncio.sleep(0.1)  # Small delay for streaming effect

                asyncio.run(stream_unstructured_response())

        except Exception as e:
            st.error(f"An error occurred: {e}")
