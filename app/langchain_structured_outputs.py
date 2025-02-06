from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from app.config import settings
from app.model import llm
from app.utils import load_schema, time_function


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
        # First pass: get the basic response (which should contain the objective type)
        initial_response, _ = generate_structured_response(user_input)
        # Check if we got an error from the fallback
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

        # TODO: This current feature breaks PRO. Maybe add `cur_time` AFTER user_input (not before).
        """
        TODO: Validate on this PRO prompt
        Create a PeriodicRevisitObjective for targets 12225,68887 using sensors RME05,LMNT06.
        Set S marking, TEST mode, priority 2, patience minutes 30, ignore other objective
        submissions false. Start at 2024-06-21 19:20:00+00:00. Set optimal frames per hour 400,
        number of frames 5, integration time 2 seconds.
        """
        # cur_time = get_current_iso_time()
        # user_input = f"Note: The current time right now is: {cur_time}. \n" + user_input

        detailed_response, _ = generate_structured_response(user_input)
        # Check if the detailed generation returned an error.
        if isinstance(detailed_response, dict) and "error" in detailed_response:
            # Reset the schema before returning the error.
            settings.update_schema(original_schema)
            return detailed_response["error"]

        # Reset the schema to the original.
        settings.update_schema(original_schema)
        # Add the objective_name field explicitly.
        detailed_response.objective_name = str(objective_type)
        assert detailed_response.objective_name == str(objective_type)
        return detailed_response
    except Exception as e:
        return f"Unexpected error during objective generation: {str(e)}"
