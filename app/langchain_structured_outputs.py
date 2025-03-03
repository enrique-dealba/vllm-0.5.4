import logging

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from app.config import settings
from app.interpretability_analysis import (
    activation_dict,
    compute_histogram,
    neuron_grad_dict,
    register_llama_hooks,
)
from app.model import llm
from app.utils import load_schema, time_function

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@time_function
def generate_structured_response(user_input: str):
    LLMResponseSchema = load_schema()

    parser = PydanticOutputParser(pydantic_object=LLMResponseSchema)

    template = """You are a helpful AI assistant that always responds in valid JSON format.
    
Format your response according to this schema:
{format_instructions}

Remember:
1. Your response MUST be valid JSON
2. Do not include any explanatory text outside the JSON
3. Ensure all required fields are included
4. Use the exact field names specified

User Query: {query}

JSON Response:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    try:
        result = chain.invoke({"query": user_input})
        return result
    except Exception as e:
        error_message = f"Error in structured response generation: {str(e)}"
        # Return a plain dictionary with an error message.
        return {"error": error_message}, 0.0


@time_function
def generate_objective_response(user_input: str):
    """Two-step process for objective generation.
    First, determine the objective type. Then, update the schema and generate details.
    If any step fails, return an error message.
    """
    try:
        initial_response, _ = generate_structured_response(user_input)
        if isinstance(initial_response, dict) and "error" in initial_response:
            return initial_response["error"]

        if not hasattr(initial_response, "objective_name"):
            return f"OBJECTIVE TYPE ERROR - Found: {initial_response}"

        objective_type = initial_response.objective_name
        detailed_objective_types = [
            "CatalogMaintenanceObjective",
            "PeriodicRevisitObjective",
            "SearchObjective",
            "DataEnrichmentObjective",
            "GeodssRevisitObjective",
            "SensorCheckoutObjective",
            "SingleIntentObjective",
            "UctObservationObjective",
            "BaselineAutonomyObjective",
        ]
        if objective_type not in detailed_objective_types:
            return f"ERROR - Found mismatched objective: '{objective_type}'."

        # Save the original schema and update it to the specific objective type.
        original_schema = settings.LLM_RESPONSE_SCHEMA
        settings.update_schema(objective_type)

        detailed_response, _ = generate_structured_response(user_input)
        if isinstance(detailed_response, dict) and "error" in detailed_response:
            settings.update_schema(original_schema)
            return detailed_response["error"]

        settings.update_schema(original_schema)
        detailed_response.objective_name = str(objective_type)
        assert detailed_response.objective_name == str(objective_type)
        return detailed_response
    except Exception as e:
        return f"Unexpected error during objective generation: {str(e)}"


def generate_structured_response_with_tracking(user_input: str):
    """Tracks neural activity during the LLM call.
    Temporarily replaces the LLM's invoke method with a tracking version that registers hooks
    on the underlying model. After the chain call, it restores the original invoke method and
    returns both the chain result and a tracking summary.
    """
    LLMResponseSchema = load_schema()
    parser = PydanticOutputParser(pydantic_object=LLMResponseSchema)
    template = (
        "You are a helpful AI assistant that always responds in valid JSON format.\n\n"
        "Format your response according to this schema:\n"
        "{format_instructions}\n\n"
        "Remember:\n"
        "1. Your response MUST be valid JSON\n"
        "2. Do not include any explanatory text outside the JSON\n"
        "3. Ensure all required fields are included\n"
        "4. Use the exact field names specified\n\n"
        "User Query: {query}\n\n"
        "JSON Response:"
    )
    prompt = PromptTemplate(
        template=template,
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser

    if not hasattr(llm, "invoke"):
        logger.error("LLM object does not have an 'invoke' method!")
        raise AttributeError("LLM object does not have an 'invoke' method!")

    original_invoke = llm.invoke
    logger.info("Original llm.invoke: %s", original_invoke)

    underlying_model = (
        llm.client.llm_engine.model_executor.driver_worker.model_runner.model
    )

    def tracked_invoke(*args, **kwargs):
        logger.info("Tracked invoke called with args: %s, kwargs: %s", args, kwargs)
        activation_dict.clear()
        neuron_grad_dict.clear()
        logger.info("Registering hooks on underlying model: %s", underlying_model)
        hook_handles = register_llama_hooks(underlying_model)
        result = original_invoke(*args, **kwargs)
        for handle in hook_handles:
            handle.remove()
        logger.info("Tracked invoke returning result: %s", result)
        return result

    object.__setattr__(llm, "invoke", tracked_invoke)
    logger.info("LLM.invoke replaced with tracked_invoke.")

    try:
        result = chain.invoke({"query": user_input})
    except Exception as e:
        error_message = f"Error in tracked structured response generation: {str(e)}"
        logger.exception(error_message)
        result = {"error": error_message}
    finally:
        object.__setattr__(llm, "invoke", original_invoke)
        logger.info("LLM.invoke restored to original.")

    tracking_data = {
        "metadata": {
            "input_text": user_input,
            "format_instructions": str(parser.get_format_instructions()),
            "full_prompt": str(prompt),
            "final_output": str(result),
            "vllm_langchain_pipeline": True,
        },
        "activations": {
            layer: {
                "summary": {
                    "mean": float(activation_dict[layer].float().mean()),
                    "std": float(activation_dict[layer].float().std()),
                    "min": float(activation_dict[layer].float().min()),
                    "max": float(activation_dict[layer].float().max()),
                },
                "histogram": compute_histogram(
                    activation_dict[layer].float().numpy().flatten(), bins=20
                ),
            }
            for layer in activation_dict
        },
        "gradients": {
            layer: {
                "summary": {
                    "mean": float(neuron_grad_dict[layer].float().mean())
                    if neuron_grad_dict.get(layer) is not None
                    else None,
                    "std": float(neuron_grad_dict[layer].float().std())
                    if neuron_grad_dict.get(layer) is not None
                    else None,
                    "min": float(neuron_grad_dict[layer].float().min())
                    if neuron_grad_dict.get(layer) is not None
                    else None,
                    "max": float(neuron_grad_dict[layer].float().max())
                    if neuron_grad_dict.get(layer) is not None
                    else None,
                },
                "histogram": compute_histogram(
                    neuron_grad_dict[layer].float().numpy().flatten(), bins=20
                )
                if neuron_grad_dict.get(layer) is not None
                else None,
            }
            for layer in neuron_grad_dict
        },
    }

    return result, tracking_data


def generate_objective_response_with_tracking(user_input: str):
    """Two-step process for objective generation.
    First, determine the objective type. Then, update the schema and generate details.
    If any step fails, return an error message.
    """
    try:
        initial_response, data_1 = generate_structured_response_with_tracking(
            user_input
        )

        data_1 = data_1 or {}

        if isinstance(initial_response, dict) and "error" in initial_response:
            return {
                "detailed_response": None,
                "error": initial_response["error"],
                "part_1": data_1,
                "part_2": {},
            }

        if not hasattr(initial_response, "objective_name"):
            error_msg = f"OBJECTIVE TYPE ERROR - Found: {str(initial_response)}"
            logger.error(error_msg)
            return {
                "detailed_response": None,
                "error": error_msg,
                "part_1": data_1,
                "part_2": {},
            }

        objective_type = initial_response.objective_name
        detailed_objective_types = [
            "CatalogMaintenanceObjective",
            "PeriodicRevisitObjective",
            "SearchObjective",
            "DataEnrichmentObjective",
            "GeodssRevisitObjective",
            "SensorCheckoutObjective",
            "SingleIntentObjective",
            "UctObservationObjective",
            "BaselineAutonomyObjective",
        ]

        if objective_type not in detailed_objective_types:
            error_msg = f"ERROR - Found mismatched objective: '{objective_type}'."
            logger.error(error_msg)
            return {
                "detailed_response": None,
                "error": error_msg,
                "part_1": data_1,
                "part_2": {},
            }

        original_schema = settings.LLM_RESPONSE_SCHEMA

        try:
            settings.update_schema(objective_type)

            detailed_response, data_2 = generate_structured_response_with_tracking(
                user_input
            )

            data_2 = data_2 or {}

            settings.update_schema(original_schema)

            if isinstance(detailed_response, dict) and "error" in detailed_response:
                return {
                    "detailed_response": None,
                    "error": detailed_response["error"],
                    "part_1": data_1,
                    "part_2": data_2,
                }

            try:
                detailed_response.objective_name = str(objective_type)
            except AttributeError as e:
                return {
                    "detailed_response": None,
                    "error": f"Could not set objective_name: {str(e)}",
                    "part_1": data_1,
                    "part_2": data_2,
                }

            return {
                "detailed_response": detailed_response,
                "part_1": data_1,
                "part_2": data_2,
            }

        except Exception as schema_error:
            try:
                settings.update_schema(original_schema)
            except:
                pass

            error_msg = f"Schema handling error: {str(schema_error)}"
            logger.exception(error_msg)
            return {
                "detailed_response": None,
                "error": error_msg,
                "part_1": data_1,
                "part_2": {},
            }

    except Exception as e:
        error_msg = f"Unexpected error during objective generation: {str(e)}"
        logger.exception(error_msg)
        return {
            "detailed_response": None,
            "error": error_msg,
            "part_1": {},
            "part_2": {},
        }
