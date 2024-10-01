import logging
import os
import sys
import time

import streamlit as st
from langchain.callbacks import StdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.smith import RunEvalConfig
from vllm import SamplingParams

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import settings
from app.model import image, llm, vlm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LangSmith tracing
handler = StdOutCallbackHandler()
tracer = LangChainTracer(project_name=settings.LANGCHAIN_PROJECT)
callback_manager = CallbackManager([handler, tracer])

st.title("LangChain LLM/VLM Interface")

# Input text
user_input = st.text_input("Enter your query:", "")

if st.button("Generate"):
    if not user_input:
        st.warning("Please enter a query.")
    else:
        start_time = time.time()
        try:
            if settings.MODEL_TYPE.upper() == "LLM":
                if llm is None:
                    st.error("LLM is not available.")
                else:
                    # Use existing callbacks if available
                    callbacks = llm.callbacks or [callback_manager]
                    generated_text = llm.invoke(user_input, callbacks=callbacks)
            elif settings.MODEL_TYPE.upper() == "VLM":
                if vlm is None:
                    st.error("VLM is not available.")
                else:
                    prompt = f"USER: <image>\n{user_input}\nASSISTANT:"
                    sampling_params = SamplingParams(
                        temperature=settings.TEMPERATURE, max_tokens=settings.MAX_TOKENS
                    )
                    outputs = vlm.generate(
                        [
                            {
                                "prompt": prompt,
                                "multi_modal_data": {"image": image},
                            }
                        ],
                        sampling_params,
                    )
                    generated_text = outputs[0].outputs[0].text
            else:
                st.error("Invalid MODEL_TYPE configuration.")
                generated_text = ""

            end_time = time.time()
            execution_time = end_time - start_time

            if generated_text:
                st.subheader("Response:")
                st.write(generated_text)
                st.write(f"Execution Time: {execution_time:.4f} seconds.")

                # Log the run to LangSmith
                run_eval_config = RunEvalConfig(
                    evaluators=["criteria", "embedding_distance"],
                    custom_evaluators=[],
                )
                tracer.on_chain_end(
                    {"response": generated_text},
                    run_id=None,
                    parent_run_id=None,
                    inputs={"query": user_input},
                    outputs={"response": generated_text},
                    tags=["streamlit_ui"],
                    metadata={
                        "execution_time": execution_time,
                        "model_type": settings.MODEL_TYPE,
                    },
                )

        except Exception as e:
            logger.exception(f"Error during generation: {e}")
            st.error(f"An error occurred: {e}")
