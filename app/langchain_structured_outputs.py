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
        return chain.invoke({"query": user_input})
    except Exception:
        # Fallback to basic response
        basic_response = LLMResponseSchema(
            response=f"I apologize, but I encountered an error processing your request. Here's my response: {llm.invoke(user_input)}",
            confidence=0.5,
        )
        return basic_response


@time_function
def generate_objective_response(user_input: str):
    """Two-step process for objective generation"""
    # First pass - get objective type
    initial_response, _ = generate_structured_response(user_input)

    # If we got an objective type that needs details, do second pass
    if hasattr(initial_response, "objective_name"):
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

        original_schema = settings.LLM_RESPONSE_SCHEMA
        settings.update_schema(objective_type)
        detailed_response, _ = generate_structured_response(user_input)
        # Reset schema to original to enable back-and-forth
        settings.update_schema(original_schema)

        # Add objective_name field
        detailed_response.objective_name = str(objective_type)
        assert detailed_response.objective_name == str(objective_type)
        return detailed_response

    return f"OBJECTIVE TYPE ERROR - Found: {initial_response}"
