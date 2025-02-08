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

    # For MockLLM: bypass actual LLM logic and return a fixed response.
    if hasattr(llm, "mock_response_type"):
        mock_response = {"objective_name": "SearchObjective"}
        return mock_response

    prompt = PromptTemplate(
        template=template,
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    try:
        chain = prompt | llm | parser
        result = chain.invoke({"query": user_input})
        return result
    except Exception as e:
        error_message = f"Error in structured response generation: {str(e)}"
        return {"error": error_message}


@time_function
def generate_objective_response(user_input: str):
    """Two-step process for objective generation.
    First, determine the objective type. Then, update the schema and generate details.
    If any step fails, return an error message.
    """
    try:
        # For MockLLM: return a fixed detailed mock response.
        if hasattr(llm, "mock_response_type"):
            mock_response = {
                "objective_name": "SearchObjective",
                "binning": None,
                "classification_marking": "U",
                "collect_request_type": "MOCK_REQUEST",
                "data_mode": "MOCK",
                "end_time_offset_minutes": 42,
                "frame_overlap_percentage": 0.5,
                "frame_type": "MOCK",
                "target_id": f"mock-uuid-{user_input}",
            }
            return mock_response

        # First pass: get the basic response (which should contain the objective type)
        initial_response, _ = generate_structured_response(user_input)
        # If an error occurred during structured generation, return an error dict.
        if isinstance(initial_response, dict) and "error" in initial_response:
            return {"error": initial_response["error"]}

        # Ensure that the basic response has the required "objective_name" field.
        if not hasattr(initial_response, "objective_name"):
            return {"error": f"OBJECTIVE TYPE ERROR - Found: {initial_response}"}

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
            return {"error": f"ERROR - Found mismatched objective: '{objective_type}'."}

        # Save the original schema and update it to the specific objective type.
        original_schema = settings.LLM_RESPONSE_SCHEMA
        settings.update_schema(objective_type)

        # (Optional) If you need to include current time or modify the query, do it here.
        # For example:
        # cur_time = get_current_iso_time()
        # user_input = f"Note: The current time right now is: {cur_time}. \n" + user_input

        detailed_response, _ = generate_structured_response(user_input)
        # If the detailed generation returned an error, reset the schema and return the error.
        if isinstance(detailed_response, dict) and "error" in detailed_response:
            settings.update_schema(original_schema)
            return {"error": detailed_response["error"]}

        # Reset the schema to the original.
        settings.update_schema(original_schema)
        # Add (or overwrite) the objective_name field explicitly.
        detailed_response.objective_name = str(objective_type)
        assert detailed_response.objective_name == str(objective_type)
        return detailed_response
    except Exception as e:
        return {"error": f"Unexpected error during objective generation: {str(e)}"}
