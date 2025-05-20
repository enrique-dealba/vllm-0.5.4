import logging

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from app.config import settings
from app.model import llm
from app.utils import load_schema, time_function

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
    original_schema = settings.LLM_RESPONSE_SCHEMA
    try:
        # First pass: get the basic response (which should contain the objective type)
        initial_response, _ = generate_structured_response(
            user_input
        )  # Uses original_schema via load_schema()

        if isinstance(initial_response, dict) and "error" in initial_response:
            # Error from the first pass, schema hasn't changed yet.
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

        # Update the schema to the specific objective type for the second pass.
        settings.update_schema(objective_type)

        # TODO: This current feature breaks PRO. Maybe add `cur_time` AFTER user_input (not before).
        """
        TODO: Validate on this PRO prompt
        Create a PeriodicRevisitObjective for targets 12225,68887 using sensors RME05,LMNT06.
        Set S marking, TEST mode, priority 2, patience minutes 30, ignore other objective
        submissions false. Start at 2024-06-21 19:20:00+00:00. Set optimal frames per hour 400,
        number of frames 5, integration time 2 seconds.
        """
        # cur_time = get_current_iso_time()
        # user_input_modified = f"Note: The current time right now is: {cur_time}. \n" + user_input
        # detailed_response_obj, _ = generate_structured_response(user_input_modified)

        detailed_response_obj, _ = generate_structured_response(
            user_input
        )  # Uses new schema via load_schema()

        if isinstance(detailed_response_obj, dict) and "error" in detailed_response_obj:
            # Error from the second pass. Schema will be reset in finally.
            return detailed_response_obj["error"]

        # Reset the schema to the original - MOVED TO FINALLY
        # settings.update_schema(original_schema) # No longer here

        # Add the objective_name field explicitly.
        # Ensure the response object is mutable if it's a Pydantic model; might need .copy() or ensure __setattr__ works
        if hasattr(detailed_response_obj, "objective_name"):
            detailed_response_obj.objective_name = str(objective_type)
        else:
            # If it's a dict or other type, handle appropriately or reconsider this logic
            # This might happen if generate_structured_response doesn't always return Pydantic model on success
            logger.warning(
                f"Cannot set objective_name on response type: {type(detailed_response_obj)}"
            )

        # assert detailed_response_obj.objective_name == str(objective_type) # Might fail if above set fails
        return detailed_response_obj
    except Exception as e:
        logger.exception(
            f"Unexpected error during objective generation for query '{user_input}': {str(e)}"
        )
        # Schema will be reset in finally.
        return f"Unexpected error during objective generation: {str(e)}"
    finally:
        # CRITICAL: Always reset the schema to its state before this function call
        settings.update_schema(original_schema)
